CUDA_VISIBLE_DEVICES=1,3 python3 infer.py --c /mlcv/WorkingSpace/Personals/thuannh/VAIPE2022/fgvc-pim/configs/evalv1.yaml \
        --csv_path /mlcv/WorkingSpace/Personals/hungcv/temp/pipeline/yolov7/results.csv   \
        --input_root /mlcv/WorkingSpace/Personals/hungcv/temp/output/detect/exp/crop_output    \
        --ocr_root /mlcv/WorkingSpace/Personals/hungcv/BKAI_VAIPE/OCR/output/exp/labels \
        --output /mlcv/WorkingSpace/Personals/thuannh/VAIPE2022/fgvc-pim/output.csv

# CUDA_VISIBLE_DEVICES=5 python3 infer_embedding.py --c ../config/infer_108class.yaml \
#         --csv_path /mlcv/WorkingSpace/Personals/hungcv/temp/pipeline/yolov7/results.csv   \
#         --input_root /mlcv/WorkingSpace/Personals/hungcv/temp/output/detect/exp/crop_output       \
#         --ocr_root /mlcv/WorkingSpace/Personals/hungcv/BKAI_VAIPE/OCR/output/exp/labels \
#         --output /mlcv/WorkingSpace/Personals/hungcv/temp/output/output.csv