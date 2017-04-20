import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn_inceptionresnetv2_imgsize1000 import FasterRCNN
from faster_rcnn.utils.timer import Timer


def test():
    import os
    # im_file = 'demo/004545.jpg'
    # im_file = 'data/VOCdevkit2007/VOC2007/JPEGImages/009036.jpg'
    im_file = '/disk2/data/ILSVRC2015/DET/Data/DET/val/ILSVRC2013_val_00004599.JPEG'
    image = cv2.imread(im_file)

    model_file = '/disk2/data/pytorch_models/trained_models/inceptionresnetv2_imgsize1000/saved_model3/faster_rcnn_200000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch3/faster_rcnn_100000.h5'
    # model_file = '/media/longc/Data/models/faster_rcnn_pytorch2/faster_rcnn_2000.h5'

    classes = np.array(['__background__',\
                         'n02672831', 'n02691156', 'n02219486', 'n02419796', 'n07739125', 'n02454379',\
                         'n07718747', 'n02764044', 'n02766320', 'n02769748', 'n07693725', 'n02777292',\
                         'n07753592', 'n02786058', 'n02787622', 'n02799071', 'n02802426', 'n02807133',\
                         'n02815834', 'n02131653', 'n02206856', 'n07720875', 'n02828884', 'n02834778',\
                         'n02840245', 'n01503061', 'n02870880', 'n02879718', 'n02883205', 'n02880940',\
                         'n02892767', 'n07880968', 'n02924116', 'n02274259', 'n02437136', 'n02951585',
                         'n02958343', 'n02970849', 'n02402425', 'n02992211', 'n01784675', 'n03000684',\
                         'n03001627', 'n03017168', 'n03062245', 'n03063338', 'n03085013', 'n03793489',\
                         'n03109150', 'n03128519', 'n03134739', 'n03141823', 'n07718472', 'n03797390',\
                         'n03188531', 'n03196217', 'n03207941', 'n02084071', 'n02121808', 'n02268443',\
                         'n03249569', 'n03255030', 'n03271574', 'n02503517', 'n03314780', 'n07753113',\
                         'n03337140', 'n03991062', 'n03372029', 'n02118333', 'n03394916', 'n01639765',\
                         'n03400231', 'n02510455', 'n01443537', 'n03445777', 'n03445924', 'n07583066',\
                         'n03467517', 'n03483316', 'n03476991', 'n07697100', 'n03481172', 'n02342885',\
                         'n03494278', 'n03495258', 'n03124170', 'n07714571', 'n03513137', 'n02398521',\
                         'n03535780', 'n02374451', 'n07697537', 'n03584254', 'n01990800', 'n01910747',\
                         'n01882714', 'n03633091', 'n02165456', 'n03636649', 'n03642806', 'n07749582',\
                         'n02129165', 'n03676483', 'n01674464', 'n01982650', 'n03710721', 'n03720891',\
                         'n03759954', 'n03761084', 'n03764736', 'n03770439', 'n02484322', 'n03790512',\
                         'n07734744', 'n03804744', 'n03814639', 'n03838899', 'n07747607', 'n02444819',\
                         'n03908618', 'n03908714', 'n03916031', 'n00007846', 'n03928116', 'n07753275',\
                         'n03942813', 'n03950228', 'n07873807', 'n03958227', 'n03961711', 'n07768694',\
                         'n07615774', 'n02346627', 'n03995372', 'n07695742', 'n04004767', 'n04019541',\
                         'n04023962', 'n04026417', 'n02324045', 'n04039381', 'n01495701', 'n02509815',\
                         'n04070727', 'n04074963', 'n04116512', 'n04118538', 'n04118776', 'n04131690',\
                         'n04141076', 'n01770393', 'n04154565', 'n02076196', 'n02411705', 'n04228054',\
                         'n02445715', 'n01944390', 'n01726692', 'n04252077', 'n04252225', 'n04254120',\
                         'n04254680', 'n04256520', 'n04270147', 'n02355227', 'n02317335', 'n04317175',\
                         'n04330267', 'n04332243', 'n07745940', 'n04336792', 'n04356056', 'n04371430',\
                         'n02395003', 'n04376876', 'n04379243', 'n04392985', 'n04409515', 'n01776313',\
                         'n04591157', 'n02129604', 'n04442312', 'n06874185', 'n04468005', 'n04487394',\
                         'n03110669', 'n01662784', 'n03211117', 'n04509417', 'n04517823', 'n04536866',\
                         'n04540053', 'n04542943', 'n04554684', 'n04557648', 'n04530566', 'n02062744',\
                         'n04591713', 'n02391049'])

    detector = FasterRCNN(classes)
    network.load_net(model_file, detector)
    detector.cuda()
    detector.eval()
    print('load model successfully!')

    # network.save_net(r'/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5', detector)
    # print('save model succ')

    t = Timer()
    t.tic()
    # image = np.zeros(shape=[600, 800, 3], dtype=np.uint8) + 255
    dets, scores, classes = detector.detect(image, 0.)
    print "classes:{},scores:{}".format(classes,scores)
    runtime = t.toc()
    print('total spend: {}s'.format(runtime))

    im2show = np.copy(image)
    for i, det in enumerate(dets):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im2show, det[0:2], det[2:4], (255, 205, 51), 2)
        cv2.putText(im2show, '%s: %.3f' % (classes[i], scores[i]), (det[0], det[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
    cv2.imwrite(os.path.join('demo', 'out.jpg'), im2show)
    # cv2.imshow('demo', im2show)
    # cv2.waitKey(0)


if __name__ == '__main__':
    test()
