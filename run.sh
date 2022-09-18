test_path=$1
pill_image_path="${test_path}/pill/image"
pres_image_path="${test_path}/prescription/image"

# # Get id on Prescription
CUDA_VISIBLE_DEVICES=$2 python3 yolov7/detect_ocr.py --weights ./weights/yolov7_ocr.pt   \
        --no-trace --conf 0.2 --save-txt --img-size 640   \
         --source $pres_image_path  \
         --project /temp/ocr --weights_ocr ./weights/vietocr.pt

# # # Detect pill
cd DINO 
CUDA_VISIBLE_DEVICES=$2 python inference.py --source $pill_image_path   \
                --output '/temp/detect' \
                --thershold 0.2 \
                --model_config_path ./config/DINO/infer.py      \
                --model_checkpoint_path ../weights/checkpoint_best_regular.pth
cd ..

#Classify
cd classify 
CUDA_VISIBLE_DEVICES=$2 python infer_vote.py --c ./config/infer_108class.yaml \
        --csv_path /temp/detect/output.csv   \
        --input_root /temp/detect/crop_img       \
        --ocr_root /temp/ocr/exp/labels \
        --test_path $1  \
        --output ../output/results.csv  
cd ..
python3 tools/post_process.py --test_path $1        
cd ./output && zip results.zip results.csv

