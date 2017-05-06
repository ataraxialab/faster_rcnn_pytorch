# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from .imdb import imdb
import ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
#import ..utils.cython_bbox
import cPickle
import subprocess
import uuid
from imagenet_eval import ILSVRC_eval
from ..fast_rcnn.config import cfg

class imagenet(imdb):
    def __init__(self, image_set, year, ILSVRC_path= None):
        imdb.__init__(self, 'ILSVRC_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._ILSVRC_path = self._get_default_path() if ILSVRC_path is None \
		            else ILSVRC_path
        self._map_det_path = os.path.join(self._ILSVRC_path, 'devkit/data/map_det.txt')
        self._DET_path = os.path.join(self._ILSVRC_path,'DET')
        self._classes, self._class_to_ind, self._class_to_name \
                        = self._load_class_wnids(self._map_det_path)
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._ILSVRC_path), \
                'Path does not exist: {}'.format(self._ILSVRC_path)
        assert os.path.exists(self._map_det_path), \
                'Path does not exist: {}'.format(self._map_det_path)
        assert os.path.exists(self._DET_path), \
                'Path does not exist: {}'.format(self._DET_path)
        print "class imagenet initialization done."

    def _load_class_wnids(self, map_det_path):
        classes = ['__background__'] # always index 0
        classes_to_ind = {}  # key = WNID, value = ILSVRC2015_DET_ID
        classes_to_name = {}  # key = WNID, value = class name
        for line in open(map_det_path):
            WNID, ILSVRC2015_DET_ID, class_name = line.split(' ', 2)
            classes.append(WNID)
            classes_to_ind[WNID] = ILSVRC2015_DET_ID
            classes_to_name[WNID] = class_name

        return tuple(classes), classes_to_ind, classes_to_name


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        path = self.image_path_from_index(self._image_index[i])
        # print "{}-th image:{}".format(i, path)
        return path

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._DET_path, 'Data', 'DET', self._image_set,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._DET_path + /ImageSets/DET/train_satisfied.txt
        if self._image_set == 'train': 
            image_set_file = os.path.join(self._DET_path, 'ImageSets', 'DET',
                                      self._image_set + '10.txt')
            assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                image_index = [x.split(' ')[0] for x in f.readlines()]

        else:
            image_set_file = os.path.join(self._DET_path, 'ImageSets', 'DET',\
                                          self._image_set + '.txt')
            assert os.path.exists(image_set_file), \
                    'Path does not exist: {}'.format(image_set_file)
            with open(image_set_file) as f:
                image_index = [x.split(' ')[0] for x in f.readlines()]
        print "load image set index done"

        return image_index


    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        print 'gt_roidb cache_file:{}'.format(cache_file)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        
        print 'gt_roidb: load ILSVRC annotation'
        gt_roidb = [self._load_ILSVRC_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'imagenetdevkit' + self._year)

    def _load_ILSVRC_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the ILSVRC
        format.
        """
        filename = os.path.join(self._DET_path, 'Annotations', 'DET', self._image_set, index + '.xml')
        """
        if not os.path.exists(filename):
            print "gt_overlaps_null:{}".format(scipy.sparse.csr_matrix(np.zeros((0, self.num_classes), dtype=np.float32)))
            return {'boxes' : np.zeros((0, 4), dtype=np.uint16),
                'gt_classes': np.zeros((0), dtype=np.int32),
                'gt_overlaps' :scipy.sparse.csr_matrix(np.zeros((0, self.num_classes), dtype=np.float32)),
                'flipped' : False,
                'seg_areas' : np.zeros((0), dtype=np.float32)}
        """
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, int(cls)] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)
        print "gt_overlaps:{}".format(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}


    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id


    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


    def _write_ILSVRC_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print 'Writing {} ILSVRC results file'.format(cls)
            filename = self._get_ILSVRC_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


    def _get_ILSVRC_results_file_template(self):
        # /disk2/data/ILSVRC2015/output/detResult/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._ILSVRC_path,
            'output',
            'detResult',
            filename)
        return path


    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._DET_path,
            'Annotations',
            'DET',
            self._image_set,
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._DET_path,
            'ImageSets',
            'DET',
            self._image_set + '.txt')
        cachedir = os.path.join(self._DET_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_ILSVRC_results_file_template().format(cls)
            rec, prec, ap = ILSVRC_eval(
                 filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


    def evaluate_detections(self, all_boxes, output_dir):
        self._write_ILSVRC_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_ILSVRC_results_file_template().format(cls)
                os.remove(filename)


if __name__ == '__main__':
    from datasets.imagenet import imagenet
    d = imagenet('train', '2015', '/disk2/data/ILSVRC2015')
    res = d.roidb
    from IPython import embed; embed()
