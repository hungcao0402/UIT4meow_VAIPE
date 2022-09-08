CUDA_VISIBLE_DEVICES=0 python3 yolov7/detect_ocr.py --weights ./weights/yolov7_ocr.pt   \
        --no-trace --conf 0.2 --save-txt --img-size 640   \
         --source /mlcv/Databases/VAIPE/public_test/prescription/image  \
         --project output/ocr --weights_ocr ./weights/vietocr.pt

