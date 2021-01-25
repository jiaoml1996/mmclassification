import os.path as osp
import pydicom
import cv2
import time
import random

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes()`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=True,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 mode='train',
                 package_size=16):
        assert mode in ['train', 'val', 'test']
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.mode = mode
        self.package_size = package_size

    def process_dcm(self, dcm, bmp_arr):
        img = dcm.pixel_array
        unknow = 'unknow'
        if dcm.get('Modality', unknow) == 'CT':
            RescaleSlope = dcm.get('RescaleSlope', unknow)
            RescaleIntercept = dcm.get('RescaleIntercept', unknow)
            if RescaleSlope != unknow and RescaleIntercept != unknow:
                img = img * RescaleSlope + RescaleIntercept
        else:
            pass
        if bmp_arr is None:
            return img
        if img.shape == bmp_arr.shape:
            return img
        new_img = np.ones(bmp_arr.shape, dtype=img.dtype) * np.min(img)
        edge = (bmp_arr.shape[0] - img.shape[0]) // 2
        new_img[edge:img.shape[0]+edge] = img
        return new_img

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filenames = [osp.join(results['img_prefix'], filename) for filename in results['img_info_list']['filenames']]
        

        if self.mode == 'train':
            # 有0.5的概率随机打乱
            seed = np.random.random()
            if seed <= 0.5:
                random.shuffle(filenames)
                random.shuffle(filenames)
            else:
                new_filenames = [(int(pydicom.read_file(filename).get('InstanceNumber', 0)), filename) for filename in filenames]
                new_filenames = sorted(new_filenames, key=lambda x:x[0])
                filenames = [filename[1] for filename in new_filenames]
        else:
            # 测试模式下, 将所有序列进行排序
            new_filenames = [(int(pydicom.read_file(filename).get('InstanceNumber', 0)), filename) for filename in filenames]
            new_filenames = sorted(new_filenames, key=lambda x:x[0])
            filenames = [filename[1] for filename in new_filenames]

        # 16个filename组成一个包
        all_filenames = []
        cnt = 0
        for i in range(self.package_size):
            all_filenames.append(filenames[cnt%len(filenames)])
            cnt += 1
        imgs = []
        for filename in all_filenames:
            # 读取和处理数据
            data_dcm = pydicom.read_file(filename)
            bmp = cv2.imread(filename.replace(
                '.DCM', '.bmp'), cv2.IMREAD_GRAYSCALE)
            img = self.process_dcm(data_dcm, bmp)
            # 将图像进行窗宽窗位的转化, 转化到0-255
            center = data_dcm.get('WindowCenter', None)
            width = data_dcm.get('WindowWidth', None)
            if center != None and width != None:
                min_ = (2 * center - width) / 2.0 + 0.5
                max_ = (2 * center + width) / 2.0 + 0.5
                dFactor = 255 / (max_ - min_)
                img = (img - min_) * dFactor
                img[img < 0.0] = 0
                img[img > 255] = 255
            
            seed = np.random.random()
            if seed < 0.5:
                bbox = results['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
                raw_img = img.copy()
                patch_img = raw_img[y1:y2, x1:x2]

                self.target_size = (112, 112)
                
                patch = np.zeros(shape=self.target_size, dtype=np.float32)
                # # 将框进行向外扩展
                target_h = self.target_size[0]
                target_w = self.target_size[1]
                edge_h = target_h - (y2 - y1)
                edge_w = target_w - (x2 - x1)
                y1 = max(0, y1 - edge_h // 2)
                x1 = max(0, x1 - edge_w // 2)
                y2 = min(y2 + edge_h // 2, img.shape[0])
                x2 = min(x2 + edge_w // 2, img.shape[1])
                patch[:y2-y1, :x2-x1] = img[y1:y2, x1:x2]
                img = patch
            img = np.expand_dims(img, axis=2)
            imgs.append(img)

        # 保存数据
        # for idx, img in enumerate(imgs):
        #     cv2.imwrite('test_{}_{}.bmp'.format(idx, time.time()), img)
        # exit(0)
        img = np.concatenate(imgs, axis=2)

        # 图像归一化
        # max_ = np.max(img)
        # min_ = np.min(img)
        # img = (img - min_) / (max_ - min_)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filenames'] = all_filenames
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
