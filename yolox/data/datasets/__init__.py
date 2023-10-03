#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

from .coco import COCODataset
from .dais import DAISDataset
from .coco_classes import COCO_CLASSES
from .dais_classes import DAIS_CLASSES
from .datasets_wrapper import CacheDataset, ConcatDataset, Dataset, MixConcatDataset
from .mosaicdetection import MosaicDetection
from .voc import VOCDetection

# from .dais_xml import DAISDataset2

