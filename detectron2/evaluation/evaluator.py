# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch
from detectron2.utils import comm
from detectron2.utils.comm import is_main_process
from detectron2.config import global_cfg as cfg
from detectron2.layers.prefetcher import data_prefetcher
from detectron2.utils.comm import get_world_size


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def record_losses(self, input, output):
        pass

    def get_avg_losses(self):
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def record_losses(self, input, output):
        for evaluator in self._evaluators:
            evaluator.record_losses(input, output)

    def get_avg_losses(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.get_avg_losses()
            if is_main_process():
                for k, v in result.items():
                    assert (
                            k not in results
                    ), "Different evalutors produce results with the same key {}".format(k)
                    results[k] = v
        return results

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evalutors produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model (in eval mode) on the data_loader and evaluate the metrics with evaluator.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run

    Returns:
        The return value of `evaluator.evalute()`
    """

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} expressions".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = cfg.MODEL.PRINT_WINDOWSIZE.VIS_TEST
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()

            loss_dict, outputs = model(inputs)
            # collect the results during inference and evaluate at final.

            # This assumes we do DDP-style training, which is currently the only
            # supported method in detectron2.
            losses = sum(loss for loss in loss_dict.values())
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict['total_loss'] = losses.detach().cpu().item()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            evaluator.process(None, outputs)
            evaluator.record_losses(metrics_dict, outputs[-1])

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    losses = evaluator.get_avg_losses()
    losses.update(results)

    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if losses is None:
        losses = {}
    return losses


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

    world_size = get_world_size()
    if cfg.SOLVER.FIX_BACKBONE:
        if world_size > 1:
            try:
                model.module.backbone.eval()
                if cfg.MODEL.VG.USING_ELMO:
                    model.module.weakly_vg_head.phrase_embed.elmo.eval()
            except:
                model.backbone.eval()
                if cfg.MODEL.VG.USING_ELMO:
                    model.weakly_vg_head.phrase_embed.elmo.eval()
        else:
            model.backbone.eval()
            if cfg.MODEL.VG.USING_ELMO:
                model.weakly_vg_head.phrase_embed.elmo.eval()