# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import json
import logging
import os
from collections import defaultdict
from contextlib import contextmanager
import torch
from fvcore.common.file_io import PathManager
from fvcore.common.history_buffer import HistoryBuffer

from detectron2.config import global_cfg as cfg




training_tags = {'total_loss':'Training/Total_loss', 'noun_reconst_loss': 'Training/noun_reconst_loss', 'rel_reconst_loss': 'Training/rel_reconst_loss','noun_cst_loss': 'Training/noun_cst_loss', 'noun_topk_reconst_loss': 'Training/noun_topk_reconst_loss', 'noun_topk_s2_reconst_loss': 'Training/noun_topk_s2_reconst_loss',
                'disc_img_sent_loss_s2':"Training/disc_img_sent_loss_s2", 'disc_img_sent_loss_s3':"Training/disc_img_sent_loss_s3", 'disc_img_sent_loss_s1':"Training/disc_img_sent_loss_s1", 'disc_vis_phr_loss_s2':"Training/disc_vis_phr_loss_s2", 'disc_vis_phr_loss_s1':"Training/disc_vis_phr_loss_s1",
                'atten_cls_loss':"Training/atten_cls_loss", 'atten_cls_topk_loss':"Training/atten_cls_topk_loss", 'reg_loss_s1':"Training/reg_loss_s1", 'reg_loss_s2':"Training/reg_loss_s2", 'reg_loss_s3':"Training/reg_loss_s3",
                 'atten_cls_nodet_loss': "Training/atten_cls_nodet_loss", 'atten_cls_topk_nodet_loss': "Training/atten_cls_topk_nodet_loss", 'noun_topk2_reconst_loss': 'Training/noun_topk2_reconst_loss',
                'data_time': 'zTime/data_time', 'vis_consistency_loss':'Training/vis_consistency_loss', 'disc_img_sent_loss_pixel': 'Training/disc_img_sent_loss_pixel', 'noun_pixel_reconst_loss': 'Training/noun_pixel_reconst_loss',
                 'attr_loss_s1':"Training/attr_loss_s1", 'attr_loss_s2':"Training/attr_loss_s2", 'rel_const_loss_s1':'Training/rel_const_loss_s1', 'rel_const_loss':'Training/rel_const_loss',
                 'rel_cls_loss_s1':'Training/rel_cls_loss_s1', 'rel_cls_loss':'Training/rel_cls_loss', 'sem_nouns_cls_loss_s1': 'Training/sem_nouns_cls_loss_s1', 'sem_nouns_cls_loss': 'Training/sem_nouns_cls_loss',
                 'sem_attrs_cls_loss_s1': 'Training/sem_attrs_cls_loss_s1', 'sem_attrs_cls_loss': 'Training/sem_attrs_cls_loss', "sub_loss_s0":"Training/sub_loss_s0",
                 "sub_loss_s1":"Training/sub_loss_s1", "obj_loss_s0":"Training/obj_loss_s0", "obj_loss_s1":"Training/obj_loss_s1", "rel_loss_s0":"Training/obj_loss_s0", "rel_loss_s1":"Training/obj_loss_s1",
                 "attr_loss_s0": "Training/obj_loss_s0"}


val_tags = {'flickr30k_val':{'total_loss':'Validation/Total_loss', 'noun_reconst_loss': 'Validation/Noun_reconst_loss', 'acc': "Validation/acc@top1", "acc_topk": "Validation/acc@topk"},
            'flickr30k_test':{'total_loss':'Validation/Total_loss', 'noun_reconst_loss': 'Validation/Noun_reconst_loss', 'acc': "Validation/acc@top1", "acc_topk": "Validation/acc@topk"}}

_CURRENT_STORAGE_STACK = []


def get_event_storage():
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]


class JSONWriter:
    """
    Write scalars to a json file.
    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:

    .. code-block:: none

        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...

    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = PathManager.open(json_file, "a")
        # self._window_size = window_size
        self._window_size = cfg.MODEL.PRINT_WINDOWSIZE.VIS_TRAIN

    def write(self):
        storage = get_event_storage()
        to_save = {"iteration": storage.iter}
        to_save.update(storage.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def __del__(self):
        # not guaranteed to be called at exit, but probably fine
        self._file_handle.close()


class TensorboardXWriter:
    """
    Write all scalars to a tensorboard file.
    """

    def __init__(self, log_dir: str, window_size: int = 20, **kwargs):
        """
        Args:
            log_dir (str): The directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = cfg.MODEL.PRINT_WINDOWSIZE.VIS_TRAIN
        # from tensorboardX import SummaryWriter

        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir, **kwargs)

    def write(self):
        storage = get_event_storage()
        for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
            self._writer.add_scalar(k, v, storage.iter)

    def __del__(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class CommonMetricPrinter:
    """
    Print __common__ metrics to the terminal, including
    iteration time, ETA, memory, all losses, and the learning rate.

    To print something different, please implement a similar printer by yourself.
    """

    def __init__(self, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter
        self._window_size = cfg.MODEL.PRINT_WINDOWSIZE.VIS_TRAIN

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter

        data_time, time = None, None
        eta_string = "N/A"
        try:
            data_time = storage.history("zTime/data_time").avg(self._window_size)
            time = storage.history("zTime/time").global_avg()
            eta_seconds = storage.history("zTime/time").median(1000) * (self._max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass

        try:
            lr = "{:.6f}".format(storage.history("zTime/LR").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        # This are not saved in training logs
        self.logger.info(
            """\
eta:{eta} iter:{iter} {losses} {MeanIoU}
{time} {data_time} lr:{lr} {memory}\
""".format(
                eta=eta_string,
                iter=iteration,
                losses=" ".join(
                    [
                        "{}:{:.4f}".format(k.replace('Training/', ''), v.avg(self._window_size))
                        for k, v in storage.histories().items()
                        if "Training" in k and ('loss' in k or 'Loss' in k)
                    ]
                ),
                MeanIoU=" ".join(
                    [
                        "{}:{:.4f}".format(k.replace('Training/', ''), v.avg(self._window_size))
                        for k, v in storage.histories().items()
                        if "Training" in k and ('IoU' in k or 'iou' in k)
                    ]
                ),
                time="time:{:.4f}".format(time) if time is not None else "",
                data_time="data_time:{:.4f}".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem:{:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
            )
        )


class EventStorage:
    """
    The user-facing class that provides metric storage functionalities.

    In the future we may add support for storing / logging other types of data if needed.
    """

    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter = start_iter
        self._current_prefix = ""

    def put_scalar(self, name, value, smoothing_hint=True):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
        """
        name = self._current_prefix + name
        history = self._history[name]
        value = float(value)
        history.update(value, self._iter)
        self._latest_scalars[name] = value

        existing_hint = self._smoothing_hints.get(name)
        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint

    def put_scalars(self, *, smoothing_hint=True, **kwargs):
        """
        Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[name -> number]: the scalars that's added in the current iteration.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.
        """
        result = {}
        window_size = cfg.MODEL.PRINT_WINDOWSIZE.VIS_TRAIN
        for k, v in self._latest_scalars.items():
            result[k] = self._history[k].avg(window_size) if self._smoothing_hints[k] else v
        return result

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should call this function at the beginning of each iteration, to
        notify the storage of the start of a new iteration.
        The storage will then be able to associate the new data with the
        correct iteration number.
        """
        self._iter += 1
        self._latest_scalars = {}

    @property
    def iter(self):
        return self._iter

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix
