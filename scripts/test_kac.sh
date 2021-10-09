output_dir="./outputs/flickr30k_kac"
DATE=`date "+%m-%d-%h"`

## Disc(smean, BN, oneEmb, inner, weighted, margin0.2)
## LR_SCHEDULER_NAME "WarmupMultiStepLR"， "WarmupPolyLR"
## Optimizer "SGD", "Adam"
## Reg(5000,1layer,0p6,s10,0p1,gap0p3)
## Phrase: "Mean", "Sum", "Last"

##

exp_name="06-17-Jun-GroundR-Visual(50-P7,DetSkipPrior,attvis,BN,spnorm)_Phr(Sent,UniMean)_visconst(w10,sum)_loss(m0p5)_SGD_0.0005_v1"


export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=3,5,6,7
python tools/train_kac.py --num-gpus 1 --eval-only --config-file configs/WeaklyGrounding-RN101-C4.yaml \
       OUTPUT_DIR "$output_dir/$exp_name"\
       SOLVER.OPTIMIZER 'SGD' \
       SOLVER.IMS_PER_BATCH 1 \
       SOLVER.DISC_IMG_SENT_LOSS False \
       SOLVER.BASE_LR 0.0005 \
       SOLVER.LR_SCHEDULER_NAME "WarmupMultiStepLR" \
       SOLVER.STEPS "(32000, 40000,)" \
       SOLVER.MAX_ITER 80000 \
       SOLVER.CHECKPOINT_PERIOD 2500 \
       MODEL.VG.SPATIAL_FEAT False \
       MODEL.VG.NETWORK 'Kac' \
       MODEL.VG.PHRASE_SELECT_TYPE 'Mean' \
       MODEL.VG.PRECOMP_TOPK 50 \
       MODEL.VG.EMBEDDING_SOURCE 'Sent' \
       MODEL.VG.LSTM_BIDIRECTION False \
       MODEL.VG.USING_DET_KNOWLEDGE True \
       MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 7 \
       DATALOADER.NUM_WORKERS 8 \
       TEST.EVAL_PERIOD 2500 \
       DATASETS.TEST "(\"flickr30k_val\", \"flickr30k_test\")" \
       MODEL.WEIGHTS "$output_dir/$exp_name/checkpoints/model_0054999.pth"