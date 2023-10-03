import torch
from torch.utils.data import Dataset
from torchvision import transforms
from loguru import logger


#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os

import cv2
import numpy as np

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img
from .dais_classes import DAIS_CLASSES

import xml.etree.ElementTree as ET



class DAISDataset2(CacheDataset):
    """
    DAIS dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        xml_file="annotations.xml",
        name="train",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        DAIS dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            xml_file (str): DAIS xml file name
            name (str): DAIS data name (e.g. 'train' or 'valid')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "DAIS-COCO")
        self.data_dir = data_dir
        self.xml_file = xml_file

        self.tree = ET.parse(os.path.join(self.data_dir, "annotations_xml", self.xml_file))
        self.root = self.tree.getroot()

        self._classes = DAIS_CLASSES
        self.class_ids = [i+1 for i in range(0, len(self._classes))]
        self.cats = [{'id': v, 'name': n, 'supercategory': ''} for v, n in zip(self.class_ids, self._classes)]
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_annotations()
        self.num_imgs = len(self.annotations)
        self.ids = [id for id in range(0, self.num_imgs)]

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]

        logger.info("num_imgs: {}".format(self.num_imgs))
        logger.info("class_ids: {}".format(self.class_ids))
        logger.info("cats: {}".format(self.cats))
        logger.info("_classes: {}".format(self._classes))
        logger.info("name: {}".format(self.name))
        logger.info("annotation[255]: {}".format(self.annotations[255]))
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_annotations(self):

        all_images = self.root.findall("image")
        images = [all_images[i] for i in range(0,len(all_images)) if all_images[i].attrib["subset"]==self.name]

        annotations = []
        for image_elem in images:
            file_name = image_elem.get("name")

            img_info = (int(image_elem.get("height")), int(image_elem.get("width")))

            height = img_info[0]
            width = img_info[1]

            objs = []
            count = 0
            for obj in image_elem.findall("box"):
                x1 = np.max((0, float(obj.get("xtl"))))
                y1 = np.max((0, float(obj.get("ytl"))))
                x2 = np.min((width, float(np.max((0, float(obj.get("xbr")))))))
                y2 = np.min((height, float(np.max((0, float(obj.get("ybr")))))))

                if obj.findall("attribute/[@name='occluded']")[0].text is not None and int(obj.findall("attribute/[@name='occluded']")[0].text) != 3:
                    # logger.info("attribute occluded: {}".format(obj.findall("attribute/[@name='occluded']")[0].text))

                    if x2 >= x1 and y2 >= y1:
                        obj.set("clean_bbox", [x1, y1, x2, y2])
                        objs.append(obj)
                        # logger.info("m(xbr): {}, x2: {}, m(ybr): {}, y2: {}".format(x1+np.max((0, float(obj.get("xbr")))),x2,y1+np.max((0, float(obj.get("ybr")))),y2))

                else:
                    count += 1
                    continue

            if count > 0:
                logger.info("For image {}, {} annotations were excluded".format(file_name, count))
                # logger.info("Attribute occluded: {}".format(len(obj.findall("attribute/[@name='occluded']"))))
                

            num_objs = len(objs)
            # logger.info("num_objs: {}".format(num_objs))

            res = np.zeros((num_objs, 5))
            for ix, obj in enumerate(objs):
                cls = self._classes.index(obj.get("label")) + 1
                # logger.info("cls: {}".format(cls))
                res[ix, 0:4] = obj.get("clean_bbox")
                res[ix, 4] = cls

            r = min(self.img_size[0] / height, self.img_size[1] / width)
            res[:, :4] *= r

            img_info = (height, width)
            resized_info = (int(height * r), int(width * r))

            annotations.append((res, img_info, resized_info, file_name))
        # logger.info("len(annotations): {}".format(len(annotations)))

        return annotations
    
    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id


