import argparse
import os
import time
from loguru import logger

import cv2

import torch

import json

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import DAIS_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

def visualize_gt(img_annotations, color, result_image, show_img=False):

    if len(img_annotations) != 0:
        flag = False
        for img_annotation in img_annotations:
            left = int(img_annotation["bbox"][0])
            top = int(img_annotation["bbox"][1])
            right = int(img_annotation["bbox"][2])
            bottom = int(img_annotation["bbox"][3])

            result_image = cv2.rectangle(result_image, (left,top), (left+right,top+bottom), color)
    
    else:
        flag = True
    
    if show_img:
        cv2.imshow('val_image', result_image)


    return result_image, flag


