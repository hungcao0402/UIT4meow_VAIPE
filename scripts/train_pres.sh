#train vietocr
cd vietocr
CUDA_VISIBLE_DEVICES=0 python3 train_vietocr.py
cd ..

#train yolov7
cd yolov7
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py    \
       --epoch 50 --single-cls --workers 8 --device 0,1 --sync-bn     \
       --batch-size 8 --data data/coco.yaml --img 640 640      \
       --cfg cfg/training/yolov7x.yaml --weights ''     \
       --name yolov7x --hyp data/hyp.scratch.p5.yaml