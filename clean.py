import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import scipy.sparse
import cPickle
IMAGE_ROOT = "/disk2/data/ILSVRC2015/DET/"


def get_satisfied_images():
    image_set_file = os.path.join(IMAGE_ROOT, 'ImageSets', 'DET')
    image_index = []
    bad_image_index = []
    for i in xrange(1,201):  # there are 200 image_set_file for training
        i_image_set_file = os.path.join(image_set_file, \
                                        'train_' + str(i) + '.txt')
        print 'process {}-th training txt: {}'.format(i, i_image_set_file)
        ## judge whether the image exists
        assert os.path.exists(i_image_set_file), \
            'Path does not exist: {}'.format(i_image_set_file)
        with open(i_image_set_file) as f:
            for x in f.readlines():  ## only use positive training samples
                #print x,len(x)
                if len(x)>2:
                    image_name, flag = x.split(' ')
                    if flag.strip() == '1' and image_ratio_check(image_name) == True :
                        #print "satisfied:{}-{}".format(i,image_name)
                        image_index.extend([image_name])
                    else:
                        #print "bad:{}".format(image_name)
                        bad_image_index.extend([image_name])
    return image_index, bad_image_index

def extra_axis(xml_obj):
    bbox = xml_obj.find('bndbox')
    x1 = float(bbox.find('xmin').text)
    y1 = float(bbox.find('ymin').text)
    x2 = float(bbox.find('xmax').text)
    y2 = float(bbox.find('ymax').text)
    return x1, y1, x2, y2

def image_ratio_check(index, image_ratio=[0.462,6.868],bbox_ratio=[0.117,15.5]):
    """
    if the image or bounding boxes are too large or too small,
    they need to be removed.
    [(x1,y1,x2,y2,name),(...)]
    """
    filename = os.path.join(IMAGE_ROOT, 'Annotations', 'DET', 'train', index + '.xml')
    tree = ET.parse(filename)

    size = tree.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)

    ## image width and height is not the same with xml
    image_path = os.path.join(IMAGE_ROOT, 'Data', 'DET', 'train', index + '.JPEG')
    im = Image.open(image_path)
    if im.size[0] != width or im.size[1] != height:
        print "Image size wxh {} {}x{} ".format(im.size, width, height)
        return False
    ## image width and height < 180
    if width <= 180 or height <= 180:
        print "Image size is too small! {}x{}".format(width, height)
        return False

    ## image width and height ratio is too small or large
    if width/height<image_ratio[0] or width/height>image_ratio[1]:
        return False

    objs = tree.findall('object')
    # bbox width and height ratio is too small or large
    for obj in objs:
        x1, y1, x2, y2 = extra_axis(obj)
        if y2-y1<=0 or (x2-x1)/(y2-y1)<bbox_ratio[0] or (x2-x1)/(y2-y1)>bbox_ratio[1]:
            return False

    return True

def save_list(image_index, savepath):
    print "save to {}".format(savepath)
    with open(savepath,"w") as f:
        for name in image_index:
            f.write(name)
            f.write('\n')
    f.close()

def _load_ILSVRC_annotation(index, classes_to_ind):
    filename = os.path.join(IMAGE_ROOT, 'Annotations', 'DET', 'train', index + '.xml') # xml_path
    # image = os.path.join(IMAGE_ROOT, 'Data', 'DET', 'train', index + '.JPEG') # image_path
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 201), dtype=np.float32) # ??????????????????????????????????????????????
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    # Load object bounding boxes into a data frame.

    for ix, obj in enumerate(objs):
        x1, y1, x2, y2 = extra_axis(obj)
        boxes[ix, :] = [x1, y1, x2, y2]
        cls = classes_to_ind[obj.find('name').text.lower().strip()]
        gt_classes[ix] = int(cls)
        overlaps[ix, int(cls)] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)
    print("gt_overlaps:{}".format(overlaps))

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

def _load_class_wnids(map_det_path):
    #classes = ['__background__'] # always index 0
    classes_to_ind = {}  # key = WNID, value = ILSVRC2015_DET_ID
    for line in open(map_det_path):
        WNID, ILSVRC2015_DET_ID, class_name = line.split(' ', 2)
        #classes.append(WNID)
        classes_to_ind[WNID] = ILSVRC2015_DET_ID
        #classes_to_name[WNID] = class_name

    #return tuple(classes), classes_to_ind, classes_to_name
    print "load classes to index done"
    return classes_to_ind


if __name__ == '__main__':
    classes_to_ind = _load_class_wnids(os.path.join(IMAGE_ROOT, '../devkit/data/map_det.txt'))
    image_index, bad_image_index = get_satisfied_images()
    save_list(image_index, os.path.join(IMAGE_ROOT, 'ImageSets', 'DET', 'train_satisfied1.txt'))
    save_list(bad_image_index,  os.path.join(IMAGE_ROOT, 'ImageSets', 'DET', 'train_bad1.txt'))

    gt_roidb = [_load_ILSVRC_annotation(index, classes_to_ind) for index in image_index]
    gt_roidb = [gt_roidb, image_index]

    cache_file = '_gt_roidb.pkl'
    with open(cache_file, 'wb') as fid:
        cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
