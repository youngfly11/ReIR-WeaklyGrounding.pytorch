#!/usr/bin/env bash
output_dir="./outputs/flickr30kRegRel"
DATE=`date "+%m-%d-%h"`

exp_name="07-26-Jul-GroundR-Visual(Rel_T50-NoST-P7-s5, VEmbRelu, SShare, DetSkipPrior, BN, ATTFuseDet2,decShare)_Phr(Sent,UniMean,1Emb)_Reg(Warmup75,2layer,0p6,smax,0p1,GAP0p1)_DISC(smean,sent,M0.2)_rel(1p0Cls,2Stage,MP_trans)_SGD_0.001_v1_work8"

## LR_SCHEDULER_NAME "WarmupMultiStepLR"， "WarmupPolyLR"
## Optimizer "SGD", "Adam"
## Phrase: "Mean", "Sum", "Last"


export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=3,5,6,7
python tools/train_weakly_grounding.py --num-gpus 4 --eval-only --config-file configs/WeaklyGrounding-RN101-C4.yaml \
       OUTPUT_DIR "$output_dir/$exp_name"\
       SOLVER.OPTIMIZER 'SGD' \
       SOLVER.IMS_PER_BATCH 40 \
       SOLVER.BASE_LR 0.001 \
       SOLVER.DISC_IMG_SENT_LOSS True \
       SOLVER.LR_SCHEDULER_NAME "WarmupMultiStepLR" \
       SOLVER.STEPS "(32000, 40000,)" \
       SOLVER.MAX_ITER 80000 \
       SOLVER.REG_START_ITER 7500 \
       SOLVER.CHECKPOINT_PERIOD 2500 \
       MODEL.VG.NETWORK 'RegRel'\
       MODEL.VG.SEMANTIC_NOUNS_TOPK 300 \
       MODEL.VG.SEM_NOUNS_LOSS_FACTOR 0.5 \
       MODEL.VG.SEMANTIC_ATTR_TOPK 79 \
       MODEL.VG.REL_CLS_LOSS_FACTOR 1.0 \
       MODEL.VG.REG_LOSS_FACTOR 0.1 \
       MODEL.VG.REG_IOU 0.6 \
       MODEL.VG.REG_GAP_SCORE 0.1 \
       MODEL.VG.SPATIAL_FEAT False \
       MODEL.VG.PHRASE_SELECT_TYPE 'Mean' \
       MODEL.VG.PRECOMP_TOPK 50 \
       MODEL.VG.S2_TOPK 5 \
       MODEL.VG.EMBEDDING_SOURCE 'Sent' \
       MODEL.VG.LSTM_BIDIRECTION False \
       MODEL.VG.USING_DET_KNOWLEDGE True \
       MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7 \
       MODEL.RELATION.IS_ON True \
       DATALOADER.NUM_WORKERS 8 \
       DATASETS.NAME 'flickr30k'\
       DATASETS.TEST "(\"flickr30k_val\", \"flickr30k_test\")" \
       MODEL.WEIGHTS "$output_dir/$exp_name/checkpoints/model_0074999.pth"



#export CUDA_VISIBLE_DEVICES=1,2
#python tools/train_weakly_grounding.py --num-gpus 2 --eval-only --config-file configs/WeaklyGrounding-RN101-C4.yaml \
#       OUTPUT_DIR "$output_dir/$exp_name" \
#       SOLVER.OPTIMIZER 'SGD' \
#       SOLVER.IMS_PER_BATCH 2 \
#       SOLVER.BASE_LR 0.0005 \
#       SOLVER.INIT_PARA False \
#       SOLVER.FIX_BACKBONE True \
#       SOLVER.LR_SCHEDULER_NAME "WarmupMultiStepLR" \
#       SOLVER.STEPS "(20000, 40000,)" \
#       SOLVER.MAX_ITER 80000  \
#       MODEL.VG.NETWORK 'Reg' \
#       SOLVER.CHECKPOINT_PERIOD 2500 \
#       MODEL.VG.SPATIAL_FEAT False \
#       MODEL.VG.PHRASE_SELECT_TYPE 'Mean' \
#       MODEL.VG.PRECOMP_TOPK 50 \
#       MODEL.VG.S2_TOPK 5 \
#       MODEL.VG.USING_ELMO False \
#       MODEL.VG.EMBEDDING_SOURCE 'Sent' \
#       MODEL.VG.LSTM_BIDIRECTION False \
#       MODEL.VG.USING_DET_KNOWLEDGE True \
#       MODEL.RELATION.IS_ON False \
#       MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7 \
#       MODEL.RELATION.INTRA_LAN False \
#       MODEL.WEIGHTS "$output_dir/$exp_name/checkpoints/model_0047499.pth" \
#       DATALOADER.NUM_WORKERS 4 \
#       DATASETS.NAME 'flickr30k' \
#       DATALOADER.ASPECT_RATIO_GROUPING True \
#       TEST.EVAL_PERIOD 2500
