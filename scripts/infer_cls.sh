cd classify &&
CUDA_VISIBLE_DEVICES=0 python infer_vote.py --c ./config/infer_108class.yaml \
        --csv_path ../output/detect/output.csv   \
        --input_root ../output/detect/crop_img       \
        --ocr_root ../output/ocr/exp/labels \
        --output ../output/results.csv
