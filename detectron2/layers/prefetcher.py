#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 15:30



import torch
from detectron2.config import global_cfg as cfg
import math
import numpy as np
import torch.nn.functional as F

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        self.device = torch.device('cuda')
        pixel_mean = torch.as_tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
        # pixel_std = torch.as_tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean)
        self.divisible_size = 2
        self.preload()

    def preload(self):
        try:
            ret = next(self.loader)

            self.images, self.phrase_ids, self.gt_boxes, self.precomp_bbox, self.precomp_score, self.precomp_det_label, \
            self.image_scale, self.relations, self.sentence, self.image_unique_id, self.det_label_embedding = ret

            image_sizes = [img.shape for img in self.images]
            max_size = tuple(max(s) for s in zip(*image_sizes))
            stride = self.divisible_size
            max_size = list(max_size)  # type: ignore
            max_size[-2] = int(math.ceil(max_size[-2] / stride) * stride)  # type: ignore
            max_size[-1] = int(math.ceil(max_size[-1] / stride) * stride)  # type: ignore
            max_size = tuple(max_size)

            image_sizes = np.array(image_sizes)[:, -2:].tolist()
            num_imgs = len(self.images)

            if num_imgs == 1:
                # This seems slightly (2%) faster.
                # TODO: check whether it's faster for multiple images as well
                image_size = image_sizes[0]
                padded = F.pad(self.normalizer(self.images[0]),[0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]], value=0,)
                self.images = padded.unsqueeze_(0)
            else:
                batch_shape = (num_imgs,) + max_size
                batched_imgs = self.images[0].new_full(batch_shape, 0)
                for img, pad_img in zip(self.images, batched_imgs):
                    pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(self.normalizer(img.float()))
                self.images = batched_imgs

        except StopIteration:
            self.images = None
            self.phrase_ids = None
            self.gt_boxes = None
            self.precomp_bbox = None
            self.precomp_score = None
            self.precomp_det_label = None
            self.image_scale = None
            self.relations = None
            self.sentence = None
            self.image_unique_id = None
            self.det_label_embedding = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.images = self.images.to(self.device, non_blocking=True)

            # self.next_input = self.next_input.cuda(non_blocking=True)
            # self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()


    def next(self):

        torch.cuda.current_stream().wait_stream(self.stream)
        images = self.images
        phrase_ids = self.phrase_ids
        gt_boxes = self.gt_boxes
        precomp_bbox = self.precomp_bbox
        precomp_score = self.precomp_score
        precomp_det_label = self.precomp_det_label
        image_scale = self.image_scale
        relations = self.relations
        sentence = self.sentence
        image_unique_id = self.image_unique_id
        det_label_embedding = self.det_label_embedding

        if images is not None:
            images.record_stream(torch.cuda.current_stream())

        self.preload()

        return images, phrase_ids, gt_boxes, precomp_bbox, precomp_score, precomp_det_label, image_scale, relations, sentence, image_unique_id, det_label_embedding
