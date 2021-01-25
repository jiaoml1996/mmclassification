import codecs
import os
import os.path as osp
import json

import numpy as np
import torch

from .base_dataset import BaseDataset
from .builder import DATASETS
from .utils import download_and_extract_archive, rm_suffix


@DATASETS.register_module()
class Medical(BaseDataset):

    CLASSES = [
        '0 - liangxing', '1 - exing'
    ]

    def get_cat_ids(self, idx):
        return [self.data_infos[idx]['gt_label'].item()]


    def get_weight(self):
        weights = np.asarray(self.class_weight, dtype=np.float32)
        for i in range(len(self.CLASSES)):
            w = (1 - (np.sum(weights == i) / weights.shape[0]))
            weights[weights == i] = w
        return weights.tolist()

    def load_annotations(self):
        
        # 读取json文件
        data_json = json.load(open(self.ann_file, 'r'))
        images = data_json['images']
        annotations = data_json['annotations']
        categories = data_json['categories']
        # 根据image id找到image
        images_ids = {image['id']:image for image in images}
        # 根据file_name找到image
        images_paths = {image['file_name']:image for image in images}
        # 根据categorie id找到categorie
        categories_ids = {categorie['id']:categorie for categorie in categories}
        # 根据image id找到annotation
        images_id_to_annotation = {annotation['image_id']:annotation for annotation in annotations}

        data_infos = []
        class_weight = []

        # 将患者进行合并

        data_infos_dict = {}

        for annotation in annotations:
            image_id = annotation['image_id']
            category_id = annotation['category_id']
            image = images_ids[image_id]
            image_dir = '/'.join(image['file_name'].split('/')[:-1])
            # 获取当前目录下所有的DCM图像
            images = list(filter(lambda x: '.DCM' in x, os.listdir(os.path.join(self.data_prefix, image_dir))))
            images = [os.path.join(image_dir, item) for item in images]

            bbox = [100000, 100000, -1, -1]
            gt_label_list = []
            # 计算最大的框
            for image_path in images:
                if image_path not in images_paths.keys():
                    continue
                image = images_paths[image_path]
                image_id = image['id']
                annotation = images_id_to_annotation[image_id]
                tmp_bbox = annotation['bbox']
                bbox[0] = min(bbox[0], tmp_bbox[0])
                bbox[1] = min(bbox[1], tmp_bbox[1])
                bbox[2] = max(bbox[2], tmp_bbox[2])
                bbox[3] = max(bbox[3], tmp_bbox[3])
                # 获取类别
                category_id = annotation['category_id']
                category = categories_ids[category_id]

                gt_label = list(filter(lambda x: category['name'] in x, self.CLASSES))[0]
                gt_label = self.CLASSES.index(gt_label)
                gt_label_list.append(gt_label)

            x1, y1, w, h = bbox
            if w < 1 or h < 1:
                continue
            # 获取gt_label
            gt_label = list(set(gt_label_list))
            assert len(gt_label) == 1
            gt_label = gt_label[0]

            
            info = {'img_prefix': self.data_prefix}
            info['img_info_list'] = {'filenames': images}
            info['gt_label'] = np.asarray(gt_label, dtype=np.int64) #gt_label_
            info['bbox'] = bbox

            flags = []
            for img in images:
                flag = '/'.join(img.split('/')[:-1])
                flags.append(flag)
            flags = list(set(flags))
            assert len(flags) == 1
            flag = flags[0]
            
            data_infos_dict[flag] = [info, gt_label]

        for key, val in data_infos_dict.items():
            data_infos.append(val[0])
            class_weight.append(val[1])

        self.class_weight = class_weight

        self.data_infos = data_infos
        return data_infos