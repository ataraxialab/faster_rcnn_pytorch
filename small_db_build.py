import os
import xml.etree.ElementTree as ET
import numpy as np

_DET_path = '/disk2/data/ILSVRC2015/DET/'

def load_image_set_index(trainlist_path):
    """
    Load the indexes listed in this dataset's image set file.
    """
    print "load train list"
    assert os.path.exists(trainlist_path), \
        'Path does not exist: {}'.format(trainlist_path)
    with open(trainlist_path) as f:
        trainlist = [x.strip() for x in f.readlines()]

    return trainlist


def classify_images(trainlist):
    """
    classify images based on their groundtruth object labels
    """
    class_dict = {}
    for image_index in trainlist:
        filename = os.path.join(_DET_path, 'Annotations/DET/train', image_index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)
        assert num_objs != 0, "num of objs = 0 in {}".format(filename)
        for obj in objs:
            cls = obj.find('name').text.lower().strip()
            if cls in class_dict:
                class_dict[cls].append(image_index)
            else:
                class_dict[cls]=[image_index]
            print "image_index:{}, class:{}".format(image_index,cls)

    return class_dict


def split_db(class_dict, num_of_splits = 10):
    split_image_index = {}
    for cls_name in class_dict.keys():
        cls_index = class_dict[cls_name]
        cls_index_num =len(cls_index)
        for split_index in xrange(num_of_splits):
            # for each split, images of every class is averaged
            cls_index_begin = split_index * cls_index_num / num_of_splits
            cls_index_end = (split_index + 1) * cls_index_num / num_of_splits
            print "cls_name:{}, split:{}, index:{}-{}".format(cls_name,split_index, cls_index_begin, cls_index_end)
            if cls_index_end >= cls_index_num:
                cls_index_end = cls_index_num - 1
            if split_index in split_image_index:
                split_image_index[split_index].extend(cls_index[cls_index_begin:cls_index_end])
            else:
                split_image_index[split_index] = [cls_index[cls_index_begin:cls_index_end]]

    return split_image_index


def save_train_txt(split_image_index, savepath):
    for i, image_index_per_split in enumerate(split_image_index.values()):
        split_i_savepath = os.path.join(savepath, 'train_small_'+str(i)+'.txt')
        print "save to {}".format(split_i_savepath)
        with open(split_i_savepath, 'w') as f:
            for image_index in image_index_per_split[0]:
                print image_index
                f.write(image_index)
                f.write('\n')
        f.close()


if __name__ == '__main__':
    train_txt_path = os.path.join(_DET_path, 'ImageSets/DET/train_satisfied.txt')
    trainlist = load_image_set_index(train_txt_path)
    class_dict = classify_images(trainlist)
    small_db_trainlist = split_db(class_dict, num_of_splits = 10)
    save_train_txt(small_db_trainlist, os.path.join(_DET_path, 'ImageSets/DET/'))

