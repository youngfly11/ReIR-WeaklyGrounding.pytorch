python demo/demo.py --config-file configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml \
  --input input.jpg --output ./viscocodect/input_vis.png\
  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl
