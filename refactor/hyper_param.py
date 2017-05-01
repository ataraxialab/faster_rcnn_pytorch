#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-05-01 16:26

import json
import os.path

import easydict
import numpy as np
import yaml


class HyperParamater(easydict.EasyDict):
    def __init__(self):
        super(HyperParamater, self).__init__()
        #
        # Training options
        #

        # region proposal network (RPN) or not
        self.IS_RPN = True
        self.ANCHOR_SCALES = [8, 16, 32]
        self.NCLASSES = 21

        # multiscale training and testing
        self.IS_MULTISCALE = False
        self.IS_EXTRAPOLATING = True

        self.REGION_PROPOSAL = 'RPN'

        self.NET_NAME = 'VGGnet'
        self.SUBCLS_NAME = 'voxel_exemplars'

        self.TRAIN = easydict.EasyDict()
        # Adam, Momentum, RMS
        self.TRAIN.SOLVER = 'Momentum'
        # learning rate
        self.TRAIN.WEIGHT_DECAY = 0.0005
        self.TRAIN.LEARNING_RATE = 0.001
        self.TRAIN.MOMENTUM = 0.9
        self.TRAIN.GAMMA = 0.1
        self.TRAIN.STEPSIZE = 50000
        self.TRAIN.DISPLAY = 10
        self.TRAIN.LOG_IMAGE_ITERS = 100
        self.TRAIN.OHEM = False

        # Scales to compute real features
        self.TRAIN.SCALES_BASE = (0.25, 0.5, 1.0, 2.0, 3.0)
        # self.TRAIN.SCALES_BASE = (1.0,)

        # parameters for ROI generating
        # self.TRAIN.SPATIAL_SCALE = 0.0625
        self.TRAIN.KERNEL_SIZE = 5

        # Aspect ratio to use during training
        # self.TRAIN.ASPECTS = (1, 0.75, 0.5, 0.25)
        self.TRAIN.ASPECTS = (1, )

        # Scales to use during training (can list multiple scales)
        # Each scale is the pixel size of an image's shortest side
        self.TRAIN.SCALES = (600, )

        # Max pixel size of the longest side of a scaled input image
        self.TRAIN.MAX_SIZE = 1000

        # Images to use per minibatch
        self.TRAIN.IMS_PER_BATCH = 2

        # Minibatch size (number of regions of interest [ROIs])
        self.TRAIN.BATCH_SIZE = 128

        # Fraction of minibatch that is labeled foreground (i.e. class > 0)
        self.TRAIN.FG_FRACTION = 0.25

        # Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
        self.TRAIN.FG_THRESH = 0.5

        # Overlap threshold for a ROI to be considered background (class = 0 if
        # overlap in [LO, HI))
        self.TRAIN.BG_THRESH_HI = 0.5
        self.TRAIN.BG_THRESH_LO = 0.1

        # Use horizontally-flipped images during training?
        self.TRAIN.USE_FLIPPED = True

        # Train bounding-box regressors
        self.TRAIN.BBOX_REG = True

        # Overlap required between a ROI and ground-truth box in order for that ROI to
        # be used as a bounding-box regression training example
        self.TRAIN.BBOX_THRESH = 0.5

        # Iterations between snapshots
        self.TRAIN.SNAPSHOT_ITERS = 5000

        # solver.prototxt specifies the snapshot path prefix, this adds an optional
        # infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
        self.TRAIN.SNAPSHOT_PREFIX = 'VGGnet_fast_rcnn'
        self.TRAIN.SNAPSHOT_INFIX = ''

        # Use a prefetch thread in roi_data_layer.layer
        # So far I haven't found this useful; likely more engineering work is required
        self.TRAIN.USE_PREFETCH = False

        # Normalize the targets (subtract empirical mean, divide by empirical stddev)
        self.TRAIN.BBOX_NORMALIZE_TARGETS = True
        # Deprecated (inside weights)
        # used for assigning weights for each coords (x1, y1, w, h)
        self.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        # Normalize the targets using "precomputed" (or made up) means and stdevs
        # (BBOX_NORMALIZE_TARGETS must also be True)
        self.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
        self.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
        self.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
        # faster rcnn dont use pre-generated rois by selective search
        # self.TRAIN.BBOX_NORMALIZE_STDS = (1, 1, 1, 1)

        # Train using these proposals
        self.TRAIN.PROPOSAL_METHOD = 'selective_search'

        # Make minibatches from images that have similar aspect ratios (i.e. both
        # tall and thin or both short and wide) in order to avoid wasting computation
        # on zero-padding.
        self.TRAIN.ASPECT_GROUPING = True
        # preclude rois intersected with dontcare areas above the value
        self.TRAIN.DONTCARE_AREA_INTERSECTION_HI = 0.5
        self.TRAIN.PRECLUDE_HARD_SAMPLES = True
        # Use RPN to detect objects
        self.TRAIN.HAS_RPN = True
        # IOU >= thresh: positive example
        self.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
        # IOU < thresh: negative example
        self.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
        # If an anchor statisfied by positive and negative conditions set to negative
        self.TRAIN.RPN_CLOBBER_POSITIVES = False
        # Max number of foreground examples
        self.TRAIN.RPN_FG_FRACTION = 0.5
        # Total number of examples
        self.TRAIN.RPN_BATCHSIZE = 256
        # NMS threshold used on RPN proposals
        self.TRAIN.RPN_NMS_THRESH = 0.7
        # Number of top scoring boxes to keep before apply NMS to RPN proposals
        self.TRAIN.RPN_PRE_NMS_TOP_N = 12000
        # Number of top scoring boxes to keep after applying NMS to RPN proposals
        self.TRAIN.RPN_POST_NMS_TOP_N = 2000
        # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        self.TRAIN.RPN_MIN_SIZE = 16
        # Deprecated (outside weights)
        self.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        # Give the positive RPN examples weight of p * 1 / {num positives}
        # and give negatives a weight of (1 - p)
        # Set to -1.0 to use uniform example weighting
        self.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
        # self.TRAIN.RPN_POSITIVE_WEIGHT = 0.5

        #
        # Testing options
        #

        self.TEST = easydict.EasyDict()

        # Scales to use during testing (can list multiple scales)
        # Each scale is the pixel size of an image's shortest side
        self.TEST.SCALES = (600, )

        # Max pixel size of the longest side of a scaled input image
        self.TEST.MAX_SIZE = 1000

        # Overlap threshold used for non-maximum suppression (suppress boxes with
        # IoU >= this threshold)
        self.TEST.NMS = 0.3

        # Experimental: treat the (K+1) units in the cls_score layer as linear
        # predictors (trained, eg, with one-vs-rest SVMs).
        self.TEST.SVM = False

        # Test using bounding-box regressors
        self.TEST.BBOX_REG = True

        # Propose boxes
        self.TEST.HAS_RPN = True

        # Test using these proposals
        self.TEST.PROPOSAL_METHOD = 'selective_search'

        ## NMS threshold used on RPN proposals
        self.TEST.RPN_NMS_THRESH = 0.7
        ## Number of top scoring boxes to keep before apply NMS to RPN proposals
        self.TEST.RPN_PRE_NMS_TOP_N = 6000
        # self.TEST.RPN_PRE_NMS_TOP_N = 12000
        ## Number of top scoring boxes to keep after applying NMS to RPN proposals
        self.TEST.RPN_POST_NMS_TOP_N = 300
        # self.TEST.RPN_POST_NMS_TOP_N = 2000
        # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
        self.TEST.RPN_MIN_SIZE = 16

        #
        # MISC
        #

        # The mapping from image coordinates to feature map coordinates might cause
        # some boxes that are distinct in image space to become identical in feature
        # coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
        # for identifying duplicate boxes.
        # 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
        self.DEDUP_BOXES = 1. / 16.

        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # We use the same pixel mean for all networks even though it's not exactly what
        # they were trained with
        self.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

        # For reproducibility
        self.RNG_SEED = 3

        # A small number that's used many times
        self.EPS = 1e-14

        # Root directory of project
        osp = os.path
        self.ROOT_DIR = osp.abspath(
            osp.join(osp.dirname(__file__), '..', '..'))

        # Data directory
        self.DATA_DIR = osp.abspath(osp.join(self.ROOT_DIR, 'data'))

        # Model directory
        self.MODELS_DIR = osp.abspath(
            osp.join(self.ROOT_DIR, 'models', 'pascal_voc'))

        # Name (or path to) the matlab executable
        self.MATLAB = 'matlab'

        # Place outputs under an experiments directory
        self.EXP_DIR = 'default'
        self.LOG_DIR = 'default'

        # Use GPU implementation of non-maximum suppression
        self.USE_GPU_NMS = True

        # Default GPU device id
        self.GPU_ID = 0

    @staticmethod
    def load_from_yaml(yaml_path):
        hyper_params = HyperParamater()
        with open(yaml_path, 'r') as f:
            hyper_params.__merge(easydict.EasyDict(yaml.load(f)))

        return hyper_params

    @staticmethod
    def load_from_json(json_path):
        hyper_params = HyperParamater()
        with open(json_path, 'r') as f:
            hyper_params.__merge(easydict.EasyDict(json.load(f)))

        return hyper_params

    def __merge(self, a):
        """Merge config dictionary a into config dictionary self, clobbering the
        options in self whenever they are also specified in a.
        """
        for k, v in a.iteritems():
            # a must specify keys that are in self
            if not self.has_key(k):
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(self[k])
            if old_type is not type(v):
                if isinstance(self[k], np.ndarray):
                    v = np.array(v, dtype=self[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                      'for config key: {}').format(
                                          type(self[k]), type(v), k))

            # recursively merge dicts
            if type(v) is easydict.EasyDict:
                try:
                    self.__merge(a[k], self[k])
                except:
                    print('Error under config key: {}'.format(k))
                    raise
            else:
                self[k] = v
