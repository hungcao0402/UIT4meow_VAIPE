cd yolov7 &&
CUDA_VISIBLE_DEVICES=1 python3 detect_pill.py   \
    --conf-thres 0.1  \
    --weights "../weights/yolov7_detect_pill.pt"  \
    --source /mlcv/Databases/VAIPE/public_test/pill/image/  \
    --agnostic-nms --no-trace    \
    --project ../output/detect    \
    --img-size 1280

# "/mlcv/WorkingSpace/Personals/hungcv/BKAI_VAIPE/weight/detect/best.pt"   \--nosave
#OCR
# CUDA_VISIBLE_DEVICES=2 python detect.py --weights runs/train/yolov7x4/weights/best.pt --no-trace --conf 0.3 --save-txt --img-size 640   \
#          --source /data/public_test/prescription/image/  \
#          --project ./output_ocr
