# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .cityscapes_evaluation import CityscapesEvaluator
from .coco_evaluation import COCOEvaluator
from .evaluator import DatasetEvaluator, DatasetEvaluators, inference_context, inference_on_dataset
from .lvis_evaluation import LVISEvaluator
from .panoptic_evaluation import COCOPanopticEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .sem_seg_evaluation import SemSegEvaluator
from .testing import print_csv_format, verify_results
from .recoco_evaluation_grounding import RECOCOEvaluator
from .flickr30k_evaluation_grounding import FLICKR30KEvaluator
from .flickr30k_evaluation_grounding_lite import FLICKR30KEvaluator as FLICKR30KEvaluatorLite
from .flickr30k_evaluation_grounding_v1 import FLICKR30KEvaluator as FLICKR30KEvaluatorV1
from .flickr30k_evaluation_grounding_reg import FLICKR30KEvaluator as FLICKR30KEvaluatorReg
from .flickr30k_evaluation_kac import FLICKR30KEvaluator as FLICKR30KEvaluatorKAC
from .flickr30k_evaluation_grounding_reg_ml import FLICKR30KEvaluator as FLICKR30KEvaluatorREGML


__all__ = [k for k in globals().keys() if not k.startswith("_")]
