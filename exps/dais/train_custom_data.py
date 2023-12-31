#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger

from yolox.exp import Exp as MyExp

import os
import random

# from .base_exp import BaseExp

# import wandb 

# wandb.init()

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 200
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True


        # --------------- basic config ----------------- #
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 3
        self.data_num_workers = 1
        self.input_size = (640, 640)
        self.random_size = (10, 20)
        self.test_size = (640, 640)
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        # modify 'silu' to 'relu' for deployment on DPU
        self.act = 'relu'
        self.thresh_lr_scale = 10
        self.device = torch.device('cuda')
        torch.backends.cudnn.enabled = False

        logger.info("GPU MEMORY AVAILABLE: " + str(torch.cuda.mem_get_info()))

        # --------------- dataset path config ----------------- #
        self.train_ann = 'instances_train.json'
        self.val_ann = 'instances_valid.json'

        # --------------- QAT --------------------------------#
        self.is_qat = True
        self.float_ckpt = None
        self.calib_dir = 'quantize_result'
        self.thresh_lr_scale = 10


    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models.yolox_q import YOLOX
            from yolox.models.yolo_pafpn_deploy_q import YOLOPAFPN
            from yolox.models.yolo_head_q import YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, self.width, in_channels=in_channels,
                act=self.act, depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes, self.width, in_channels=in_channels,
                act=self.act, depthwise=True
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        if self.is_qat:
            if self.float_ckpt:
                ckpt = torch.load(self.float_ckpt)
                print('Loading float model weight for QAT from {}'.format(self.float_ckpt))
                self.model.load_state_dict(ckpt["model"])
            self.model.to(self.device)
            dummy_input = torch.randn([1, 3, *self.test_size], dtype=torch.float32).to(self.device)
            self.qat_processor = QatProcessor(self.model, dummy_input, bitwidth=8, mix_bit=False, device=self.device)
            if os.path.exists(self.calib_dir):
                print('Loading calibration result for QAT initialization from {}'.format(self.calib_dir))
            else:
                self.calib_dir = ''
                print('The calib_dir: {} is not exists, we set it empty as default'.format(self.calib_dir))
            self.model = self.qat_processor.trainable_model(calib_dir=self.calib_dir)

        return self.model


    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    if isinstance(v, nn.Identity): # during nndct's QAT, 'bn' is merged to 'conv' and replaced by Identity
                        continue
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            threshold = [ 
                param for name, param in self.model.named_parameters()
                if 'threshold' in name
            ]
            # print("Threshold params: ")
            # [print(name) for name, param in self.model.named_parameters() if 'threshold' in name]
            q_param_group = { 
                'params': threshold,
                'lr': lr * self.thresh_lr_scale,
                'name': 'threshold'
            } 

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            optimizer.add_param_group(q_param_group)
            self.optimizer = optimizer

        return self.optimizer

    
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data.datasets import DAISDataset
        from yolox.data import TrainTransform

        return DAISDataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )
   
    

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Get dataloader according to cache_img parameter.
        Args:
            no_aug (bool, optional): Whether to turn off mosaic data enhancement. Defaults to False.
            cache_img (str, optional): cache_img is equivalent to cache_type. Defaults to None.
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
                None: Do not use cache, in this case cache_data is also None.
        """
        from yolox.data import (
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        # if cache is True, we will create self.dataset before launch
        # else we will create self.dataset after launch
        if self.dataset is None:
            with wait_for_the_master():
                assert cache_img is None, \
                    "cache_img must be None if you didn't create self.dataset before launch"
                self.dataset = self.get_dataset(cache=False, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": False}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
    
    def get_eval_dataset(self, **kwargs):
        from yolox.data import DAISDataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return DAISDataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="valid" if not testdev else "test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_eval_loader(self, batch_size, is_distributed, **kwargs):
        valdataset = self.get_eval_dataset(**kwargs)

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": False,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators.dais_evaluator import DAISEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed)
        evaluator = DAISEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,   
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

        

