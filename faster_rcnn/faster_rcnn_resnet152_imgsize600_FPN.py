import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_FPN import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
#from vgg16 import VGG16
from resnet_FPN import ResNet152


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()

        self.conv1 = Conv2d(256, 256, 3, same_padding=True)
        self.score_conv = Conv2d(256, 1 * 3 * 2, 1, relu=False, same_padding=False) # 1*3*2: len(self.anchor_scales)=1
        self.bbox_conv = Conv2d(256, 1 * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.cross_entropy = None
        self.los_box = None

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, features, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None, anchor_scales=[16], _feat_stride=[16]):
        rpn_conv1 = self.conv1(features)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(anchor_scales)*3*2)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                                   cfg_key, _feat_stride, anchor_scales)
        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas,
                                                im_info, _feat_stride, anchor_scales)

            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        # classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_label = rpn_data[0].view(-1)

        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        fg_cnt = torch.sum(rpn_label.data.ne(0))

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchor_scales)
        x = network.np_to_variable(x, is_cuda=True)
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride, anchor_scales)

        rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)


class FasterRCNN(nn.Module):

    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self, classes, debug=False):
        super(FasterRCNN, self).__init__()

        self.classes = classes
        self.n_classes = len(classes)

        self.fpn_feature = ResNet152('FPN')
        self.c5_conv1 = Conv2d(2048, 256, 1, same_padding=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.c4_conv1 = Conv2d(1024, 256, 1, same_padding=True)
        self.c4_conv3 = Conv2d(256, 256, 3, same_padding=True)
        self.c3_conv1 = Conv2d(512, 256, 1, same_padding=True)
        self.c3_conv3 = Conv2d(256, 256, 3, same_padding=True)
        self.c2_conv1 = Conv2d(256, 256, 1, same_padding=True)
        self.c2_conv3 = Conv2d(256, 256, 3, same_padding=True)

        self.rpn = RPN()
        self.roi_pool = RoIPool(7, 7, 1.0/16)
        self.fc1 = FC(7*7*256, 1024, relu=False)
        self.fc2 = FC(1024, 1024, relu=False)

        self.score_fc = FC(1024, self.n_classes, relu=False)
        self.bbox_fc = FC(1024, self.n_classes * 4, relu=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None

        # for log
        self.debug = debug

        if self.debug:
            print "classes:{}".format(self.classes)

    @property
    def loss(self):
        # print self.cross_entropy
        # print self.loss_box
        # print self.rpn.cross_entropy
        # print self.rpn.loss_box
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        C2_feature, C3_feature, C4_feature, C5_feature = self.fpn_feature(im_data)
        if self.debug:
            print "size:{},{},{},{},{}".format(im_data.size(),C2_feature.size(),C3_feature.size(),C4_feature.size(),C5_feature.size())
        #fpn: layer P5
        P5_feature = self.c5_conv1(C5_feature)
        if self.debug:
             print "P5_feature:{}".format(P5_feature.size())
        #fpn: layer P4
        P4_topdown_feature = self.upsample(P5_feature)
        P4_lateral_feature = self.c4_conv1(C4_feature)
        if self.debug:
            print "P4_topdown_feature:{},P4_lateral_feature:{}".format(P4_topdown_feature.size(), P4_lateral_feature.size())
        P4_topdown_feature = self._clip_feature(P4_topdown_feature, P4_lateral_feature)
        P4_feature = P4_topdown_feature + P4_lateral_feature
        P4_feature = self.c4_conv3(P4_feature)
        if self.debug:
             print "P4_feature:{}".format(P4_feature.size())
        #fpn: layer P3
        P3_topdown_feature = self.upsample(P4_feature)
        P3_lateral_feature = self.c3_conv1(C3_feature)
        P3_topdown_feature = self._clip_feature(P3_topdown_feature, P3_lateral_feature)
        if self.debug:
            print "P3_topdown_feature:{},P3_lateral_feature:{}".format(P3_topdown_feature.size(), P3_lateral_feature.size())
        P3_feature = P3_topdown_feature + P3_lateral_feature
        P3_feature = self.c3_conv3(P3_feature)
        if self.debug:
             print "P3_feature:{}".format(P3_feature.size())
        #fpn: layer P2
        P2_topdown_feature = self.upsample(P3_feature)
        P2_lateral_feature = self.c2_conv1(C2_feature)
        P2_topdown_feature = self._clip_feature(P2_topdown_feature, P2_lateral_feature)
        if self.debug:
            print "P2_topdown_feature:{},P2_lateral_feature:{}".format(P2_topdown_feature.size(), P2_lateral_feature.size())
        P2_feature = P2_topdown_feature + P2_lateral_feature
        P2_feature = self.c2_conv3(P2_feature)
        if self.debug:
             print "P2_feature:{}".format(P2_feature.size())

        P5_rois = self.rpn(P5_feature, im_info, gt_boxes, gt_ishard, dontcare_areas, anchor_scales=[16,], _feat_stride=[32,])
        P4_rois = self.rpn(P4_feature, im_info, gt_boxes, gt_ishard, dontcare_areas, anchor_scales=[8,], _feat_stride=[16,])
        P3_rois = self.rpn(P3_feature, im_info, gt_boxes, gt_ishard, dontcare_areas, anchor_scales=[4,], _feat_stride=[8,])
        P2_rois = self.rpn(P2_feature, im_info, gt_boxes, gt_ishard, dontcare_areas, anchor_scales=[2,], _feat_stride=[4,])
        if self.debug:
            print 'P5_rois:{},P4_rois:{},P3_rois:{},P2_rois:()'.format(P5_rois.size(), P4_rois.size(), P3_rois.size(), P2_rois.size())

        if self.training:
            P5_roi_data = self.proposal_target_layer(P5_rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            P5_rois = P5_roi_data[0]
            P4_roi_data = self.proposal_target_layer(P4_rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            P4_rois = P4_roi_data[0]
            P3_roi_data = self.proposal_target_layer(P3_rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            P3_rois = P3_roi_data[0]
            P2_roi_data = self.proposal_target_layer(P2_rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            P2_rois = P2_roi_data[0]

        # roi pool
        P5_cls_score, P5_cls_prob, P5_bbox_pred = self.roi_pool_basic_block(P5_feature, P5_rois)
        P4_cls_score, P4_cls_prob, P4_bbox_pred = self.roi_pool_basic_block(P4_feature, P4_rois)
        P3_cls_score, P3_cls_prob, P3_bbox_pred = self.roi_pool_basic_block(P3_feature, P3_rois)
        P2_cls_score, P2_cls_prob, P2_bbox_pred = self.roi_pool_basic_block(P2_feature, P2_rois)

        cls_prob = torch.cat([P5_cls_prob, P4_cls_prob, P3_cls_prob, P2_cls_prob])
        bbox_pred = torch.cat([P5_bbox_pred, P4_bbox_pred, P3_bbox_pred, P2_bbox_pred])
        rois = torch.cat([P5_rois, P4_rois, P3_rois, P2_rois])

        if self.training:
            P5_cross_entropy, P5_loss_box, tp1,tf1,fg_cnt1,bg_cnt1\
                              = self.build_loss(P5_cls_score, P5_bbox_pred, P5_roi_data)
            P4_cross_entropy, P4_loss_box, tp2,tf2,fg_cnt2,bg_cnt2\
                              = self.build_loss(P4_cls_score, P4_bbox_pred, P4_roi_data)
            P3_cross_entropy, P3_loss_box, tp3,tf3,fg_cnt3,bg_cnt3\
                              = self.build_loss(P3_cls_score, P3_bbox_pred, P3_roi_data)
            P2_cross_entropy, P2_loss_box, tp4,tf4,fg_cnt4,bg_cnt4\
                              = self.build_loss(P2_cls_score, P2_bbox_pred, P2_roi_data)
            self.cross_entropy = P5_cross_entropy + P4_cross_entropy + P3_cross_entropy +P2_cross_entropy
            self.loss_box = P5_loss_box + P4_loss_box + P3_loss_box + P2_loss_box
            self.tp = tp1+tp2+tp3+tp4
            self.tf = tf1+tf2+tf3+tf4
            self.fg_cnt = fg_cnt1+fg_cnt2+fg_cnt3+fg_cnt4
            self.bg_cnt = bg_cnt1+bg_cnt2+bg_cnt3+bg_cnt4

        return cls_prob, bbox_pred, rois


    def  _clip_feature(self, topdown_fea, lateral_fea):
        if topdown_fea.size() == lateral_fea.size():
            return topdown_fea
        else:
            topdown_fea = topdown_fea[:,:,0:lateral_fea.size(2),0:lateral_fea.size(3)]
            return topdown_fea


    def roi_pool_basic_block(self, feature, rois):
        pooled_features = self.roi_pool(feature, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc1(x)
        x = F.dropout(x, training = self.training)
        x = self.fc2(x)
        x = F.dropout(x, training = self.training)
        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        return cls_score, cls_prob, bbox_pred

    def build_loss(self, cls_score, bbox_pred, roi_data):
        # classification loss
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        maxv, predict = cls_score.data.max(1)
        tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
        tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:])) if bg_cnt > 0 and len(predict)-fg_cnt > 0 else 0
        fg_cnt = fg_cnt
        bg_cnt = bg_cnt

        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / (bg_cnt + 1e-4)
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)

        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return cross_entropy, loss_box, tp,tf,fg_cnt,bg_cnt

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        num_classes
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """
        rpn_rois = rpn_rois.data.cpu().numpy()
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer_py(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        rois = network.np_to_variable(rois, is_cuda=True)
        labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
        bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        cls_prob, bbox_pred, rois = self(im_data, im_info)
        # print "cls_prob:{},bbox_pred:{},rois:{}".format(cls_prob, bbox_pred,rois)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def load_from_npz(self, params):
        self.rpn.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)

if __name__ == '__main__':
    net = FasterRCNN(classes=[])
    print net
