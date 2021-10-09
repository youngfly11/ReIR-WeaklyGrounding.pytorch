#!/usr/bin/env bash
output_dir="./outputs/flickr30k"
DATE=`date "+%m-%d-%h"`

## Disc(smean, BN, oneEmb, inner, weighted, margin0.2)
## LR_SCHEDULER_NAME "WarmupMultiStepLR"， "WarmupPolyLR"
## Optimizer "SGD", "Adam"
## Reg(5000,1layer,0p6,s10,0p1,gap0p3)
## Phrase: "Mean", "Sum", "Last"
#$DATE-GroundR-Visual(T50-K5-NoST-P7, VEmbRelu, SShare, DetSkipPrior, BN, ATTFuseDet2, decShare)_Phr(Sent,UniMean,1Emb)_Reg(True75,2layerLeakly,0p6,smax,0p1,GAP0p3)_DISC(smean,sent,M0.2)_NoREL(bo)_SGD_0.001_v1
## NETWORK 'ML_Reg', 'Reg', 'PixelBox', 'Kac', 'Baseline', 'Baseline_s2'
## --gpu-check
## rel(0p1Cls,1Stage)
## rel(0p1Cls,2Stage,MP_trans)
## Reg(Warmup8w,2layer,0p6,smax,0p0,GAP0p1,lossOffNoApply)
## VEmbRelu, SShare, DetSkipPrior, BN, ATTFuseDet2,decShare

export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=3,5,6,7
python tools/train_weakly_grounding.py --num-gpus 4 --dist-url auto --config-file configs/WeaklyGrounding-RN101-C4.yaml \
       OUTPUT_DIR "$output_dir/$DATE-Visual(T50-NoST-P7-s5)_Phr(Sent,UniMean,1Emb)_DISC(smean,sent,M0.2)_Reg(Warmup75,2layer,0p6,smax,0p1,GAP0p1)_rel(1p0Cls,2Stage,MP_trans)_SGD_0.001"\
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
       TEST.EVAL_PERIOD 2500
