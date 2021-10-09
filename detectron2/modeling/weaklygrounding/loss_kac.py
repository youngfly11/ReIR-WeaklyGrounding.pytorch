import torch
from torch.nn import functional as F

from detectron2.layers.numerical_stability_softmax import numerical_stability_softmax
from detectron2.config import global_cfg as cfg
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from detectron2.structures import Boxes
from detectron2.utils.events import get_event_storage
from detectron2.layers.weighted_smooth_l1_loss import smooth_l1_loss as weighted_smooth_l1_loss



class WeaklyVGLossCompute():
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device = torch.device('cuda')
        self.reg_lambda = cfg.MODEL.VG.REG_LOSS_FACTOR

    def __call__(self, batch_phrase_mask, batch_decode_logits, batch_phrase_dec_ids, batch_pred_delta, batch_gt_delta, batch_pred_similarity):

        noun_reconst_loss = torch.zeros(1).to(self.device)
        visual_consistency_loss = torch.zeros(1).to(self.device)


        for (phr_mask, decode_logits, phrase_dec_ids, pred_delta, det_sim, gt_delta) in zip(batch_phrase_mask, batch_decode_logits, batch_phrase_dec_ids, batch_pred_delta, batch_pred_similarity, batch_gt_delta):


            ## here we ignore the first world reconstruction,
            phrase_dec_ids = torch.as_tensor(phrase_dec_ids).long().to(self.device)
            vx, vy = (phr_mask > 0).nonzero().transpose(0, 1)
            noun_reconst_loss += self.cross_entropy(decode_logits[vx, vy], phrase_dec_ids[vx, vy])

            """
            pred_delta: np*nb*4
            gt_delta: nb*4
            det_sim: np*nb
            """

            np, nb = det_sim.shape
            pred_delta = pred_delta.reshape(-1, 4)
            gt_delta = gt_delta.unsqueeze(0).repeat(np, 1, 1).reshape(-1, 4)
            gt_delta = gt_delta - 0.5
            n = torch.abs(pred_delta - gt_delta)
            cond = n < 1
            loss = torch.where(cond, 0.5 * n ** 2, n - 0.5).mean(1)
            vc_loss = det_sim.reshape(-1) * loss
            visual_consistency_loss += self.reg_lambda * vc_loss.sum()

        return noun_reconst_loss, visual_consistency_loss
