cd DINO
CUDA_VISIBLE_DEVICES=0 python inference.py --source /data/public_test/pill/image/   \
                --output '../output/detect' \
                --thershold 0.6 