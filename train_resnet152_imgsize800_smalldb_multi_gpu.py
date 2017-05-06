import collections

from datetime import datetime
import multiprocessing
import threading
import os

import torch
import torch.cuda
import torch.autograd
import torch.nn
import numpy as np

from faster_rcnn import network
from faster_rcnn.faster_rcnn_resnet152_imgsize800_multi_gpu import FasterRCNN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


# hyper-parameters
# ------------
print "initialize"
imdb_name = 'imagenet_2015_train'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = '/disk2/data/pytorch_models/resnet152-b121ed2d.pth'
output_dir = '/disk2/data/pytorch_models/trained_models/resnet152_imgsize600/saved_model3'

start_step = 0
end_step = 500000
save_model_steps = 100000
lr_decay_steps = {300000, 400000}
lr_decay = 1. / 10

rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False  # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
print "load cfg from file"
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)


class Future(object):
    def __init__(self, pipe):
        self.__pipe = pipe

    def get(self):
        return self.__pipe.recv()


# training
print "start training"


class GradUpdater(object):
    def __init__(self, count, device_id):
        self.__count = count
        self.__device_id = device_id
        self.__grad_queue = multiprocessing.Queue(count)
        self.__grads = []
        self.__suspend_trainers = []
        self.__is_closed = False

    def step(self, grads):
        p, c = multiprocessing.Pipe()
        self.__grad_queue.put((p, grads))
        return Future(c)

    def is_all_step(self):
        with self.__lock:
            return self.__count == len(self.__grads)

    def close(self):
        p, c = multiprocessing.Pipe()
        self.__grad_queue.put((p, None))
        return Future(c)

    def _calc_grad(self, gs):
        out = reduce(lambda x, y: torch.add(x, 1.0, y), gs,
                     torch.zeros(*gs[0].size()).cuda(self.__device_id))
        torch.div(out, len(gs))
        return out

    def update(self):
        pipes, grads = [], []
        while True:
            p, g = self.__grad_queue.get()
            print 'update grader recv:', p, g
            if g is None:
                break

            pipes.append(p)
            grads.append(g)

            if len(grads) < self.__count:
                continue

            name2grads = collections.defaultdict(list)
            for gs in grads:
                for name, g in gs.items():
                    g.cuda(self.__device_id)
                    name2grads[name].append(g)

            updated_grad = {
                name: self._calc_grad(gs)
                for name, gs in name2grads.items()
            }

            for p in pipes:
                p.send(updated_grad)
            pipes, grads = [], []


def net_grads(net,
              rule=lambda k: k.endswith('.bias') or k.endswith('.weight')):
    ret = {}
    for k, v in net.state_dict().items():
        if not rule(k):
            continue
        if isinstance(v, torch.cuda.FloatTensor):
            ret[k] = v
            continue
        if isinstance(v, torch.autograd.Variable):
            ret[k] = v.data.grad
        print 'unknow type', type(v)
    return ret


def update_net_grads(net, grads):
    state = net.state_dict()
    for name, s in state:
        if name not in grads:
            continue
        if isinstance(s, torch.cuda.FloatTensor):
            s.copy(grads[name])
            continue
        if isinstance(s, torch.autograd.Variable):
            s.date.grad.copy(grads[name])
            continue


def train(device_id, net, optimizer, data_layer, grad_updater, params):
    train_loss = 0
    tp, tf, fg, bg = 0., 0., 0, 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()
    lr = cfg.TRAIN.LEARNING_RATE

    to_var = lambda x: network.np_to_variable(x).cuda(device_id) if x is not None else None
    for step in range(start_step, end_step + 1):
        print "step:{}".format(step)
        # get one batch
        blobs = data_layer.forward()
        im_data = to_var(blobs['data'])
        im_info = to_var(blobs['im_info'])
        gt_boxes = to_var(blobs['gt_boxes'])
        gt_ishard = to_var(blobs['gt_ishard'])
        dontcare_areas = to_var(blobs['dontcare_areas'])

        # forward
        net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
        loss = net.loss + net.rpn.loss

        if _DEBUG:
            tp += float(net.tp)
            tf += float(net.tf)
            fg += net.fg_cnt
            bg += net.bg_cnt

        train_loss += loss.data[0]
        step_cnt += 1

        # backward
        optimizer.zero_grad()
        print threading.current_thread().name, 'backward begin'
        loss.cuda(device_id).backward()
        print threading.current_thread().name, 'backward end'
        network.clip_gradient(net, 10.)
        optimizer.step()

        grads = net_grads(net)
        print threading.current_thread().name, 'step grads'
        f = grad_updater.step(grads)
        grads = f.get()
        print threading.current_thread().name, 'update grads'
        update_net_grads(net, grads)
        print threading.current_thread().name, 'update done'

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration

            log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
                step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps)
            log_print(log_text, color='green', attrs=['bold'])

            if _DEBUG:
                log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' %
                          (tp / (fg + 1e-4) * 100., tf / (bg + 1e-4) * 100.,
                           fg / step_cnt, bg / step_cnt))
                log_print(
                    '\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f'
                    % (net.rpn.cross_entropy.data.cpu().numpy()[0],
                       net.rpn.loss_box.data.cpu().numpy()[0],
                       net.cross_entropy.data.cpu().numpy()[0],
                       net.loss_box.data.cpu().numpy()[0]))
            re_cnt = True

        if use_tensorboard and step % log_interval == 0:
            exp.add_scalar_value(
                'train_loss', train_loss / step_cnt, step=step)
            exp.add_scalar_value('learning_rate', lr, step=step)
            if _DEBUG:
                exp.add_scalar_value(
                    'true_positive', tp / fg * 100., step=step)
                exp.add_scalar_value(
                    'true_negative', tf / bg * 100., step=step)
                losses = {
                    'rpn_cls':
                    float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                    'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                    'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                    'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])
                }
                exp.add_scalar_dict(losses, step=step)

        if (step % save_model_steps == 0) and step > 0:
            save_name = os.path.join(output_dir,
                                     'faster_rcnn_{}.h5'.format(step))
            network.save_net(save_name, net)
            print('save model: {}'.format(save_name))
        if step in lr_decay_steps:
            lr *= lr_decay
            print "learning rate decay:{}".format(lr)
            optimizer = torch.optim.SGD(
                params[435:],
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay)

        if re_cnt:
            tp, tf, fg, bg = 0., 0., 0, 0
            train_loss = 0
            step_cnt = 0
            t.tic()
            re_cnt = False

    grad_updater.close()


def build_train_params(device_id):
    # load data
    print "load data"
    imdb = get_imdb(imdb_name)
    print "prepare roidb"
    rdl_roidb.prepare_roidb(imdb)
    print "done"
    roidb = imdb.roidb
    print "ROIDataLayer"
    data_layer = RoIDataLayer(roidb, imdb.num_classes)

    # load netZZ
    print "initialize faster rcnn"
    net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
    network.weights_normal_init(net, dev=0.01)
    #network.load_pretrained_npy(net, pretrained_model)
    network.load_pretrained_pth(net, pretrained_model)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    net.cuda(device_id)
    net.train()
    params = list(net.parameters())
    print params[435].size()
    # optimizer = torch.optim.Adam(params[-8:], lr=lr)
    optimizer = torch.optim.SGD(
        params[435:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    return data_layer, net, optimizer, params


def main():
    gpu_count = torch.cuda.device_count()
    grad_updater = GradUpdater(gpu_count, 0)
    trainers = []
    for i in range(gpu_count):
        data_layer, net, optimizer, params = build_train_params(i)
        trainers.append(
            multiprocessing.Process(
                target=train,
                name='trainer-%d' % i,
                args=(i, net, optimizer, data_layer, grad_updater)))

    for t in trainers:
        t.start()

    grad_updater.update()



if __name__ == '__main__':
    main()
