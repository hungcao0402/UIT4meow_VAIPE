test_path=$1
pill_image_path="${test_path}/pill/image"
pres_image_path="${test_path}/prescription/image"

# # Get id on Prescription
CUDA_VISIBLE_DEVICES=$2 python3 yolov7/detect_ocr.py --weights ./weights/yolov7_ocr.pt   \
        --no-trace --conf 0.2 --save-txt --img-size 640   \
         --source $pres_image_path  \
         --project /tempt/ocr --weights_ocr ./weights/vietocr.pt

# # Detect pill
cd DINO 
CUDA_VISIBLE_DEVICES=$2 python inference.py --source $pill_image_path   \
                --output '/tempt/detect' \
                --thershold 0.4 \
                --model_config_path ./config/DINO/infer.py      \
                --model_checkpoint_path ../weights/checkpoint_best_regular.pth
cd ..

#Classify
cd classify 
CUDA_VISIBLE_DEVICES=$2 python infer_vote.py --c ./config/infer_108class.yaml \
        --csv_path /tempt/detect/output.csv   \
        --input_root /tempt/detect/crop_img       \
        --ocr_root /tempt/ocr/exp/labels \
        --output ../output1/results.csv

cd ../output && zip results.zip results.csv
