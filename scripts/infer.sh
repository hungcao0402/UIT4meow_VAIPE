# CUDA_VISIBLE_DEVICES=$3 python3 yolov7/detect_pill.py   \
#     --conf-thres 0.1  \
#     --weights "./weights/yolov7_detect_pill.pt"  \
#     --source $pill_image_path  \
#     --agnostic-nms --no-trace    \
#     --project ./output/detect    \
#     --img-size 1280
