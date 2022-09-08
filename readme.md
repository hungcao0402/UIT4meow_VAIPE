# VAIPE: Medicine Pill Image Recognition Challenge - Team: UIT4meow solution

## Table of contents

  - [Environment Setup:](#environment-setup)
  - [Inference on testing data](#inference-on-testing-data)
  - [Train](#train)

---

## Environment Setup:
We recommend to useDocker for bng environment.
1. Build Docker Image with Dockerfile:
```
docker build -t uit4meow - < Dockerfile
```
2. Run Docker container:
```
docker run -it -v DATA_DIR:/data/:ro -v "$PWD":/workspace  --gpus all --ipc=host --name uit4meow uit4meow bash
```

Where `DATA_DIR=<path to public_test folder>`.
The directory public_test should look like this:
```text
data
|__ public_train
|     |__ pill 
|     |     |__ image
|     |           |__ VAIPE_P_0_0.jpg
|     |           |__ VAIPE_P_0_1.jpg
|     |           |__ VAIPE_P_0_2.jpg
|     |           |__ ...
|     |__ prescription
|           |__ image
|                 |__ VAIPE_P_TEST_0.png
|                 |__ VAIPE_P_TEST_1.png
|                 |__ VAIPE_P_TEST_2.png
|                 |__ ...
|__ public_valid
|     |__ pill 
|          |__ image
|                |__ VAIPE_P_0_0.jpg
|                |__ VAIPE_P_0_1.jpg
|                |__ VAIPE_P_0_2.jpg
|                |__ ...
|__ public_test
      |__ pill 
      |     |__ image
      |           |__ VAIPE_P_0_0.jpg
      |           |__ VAIPE_P_0_1.jpg
      |           |__ VAIPE_P_0_2.jpg
      |           |__ ...
      |__ prescription
            |__ image
                  |__ VAIPE_P_TEST_0.png
                  |__ VAIPE_P_TEST_1.png
                  |__ VAIPE_P_TEST_2.png
                  |__ ...
``` 

3. Install
```
pip install -r requirements.txt

# Compiling CUDA operators
cd DINO/models/dino/ops
python setup.py build install
cd ../../../..
```

---

## Inference on testing data
For inference to submit, we provide a file script to run the complete pipeline. The file results.zip to submit will be in foler output.
```
#Download weight
bash scripts/download_infer.sh

bash run.sh /data/public_test 0     # 0 is gpu-id
```

---

## Train
Train on 2 GPUs Tesla P40 24GB

<details>
      <summary>Yolov7</summary>
    
      python tools/generate_train_detect_pres.py
      cd yolov7
      CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 train.py    \
            --epoch 50 --single-cls --workers 8 --device 0,1 --sync-bn     \
            --batch-size 8 --data data/coco.yaml --img 640 640      \
            --cfg cfg/training/yolov7x.yaml --weights ''     \
            --name yolov7x --hyp data/hyp.scratch.p5.yaml
</details>
<details>
      <summary>VietOCR</summary>

      python tools/crop_pres.py
      CUDA_VISIBLE_DEVICES=0 python train_vietocr.py
</details>

<details>
      <summary>DINO</summary>

      prepare data
      ln -s /data/public_train ./dataset/dino/train2017
      ln -s /data/public_val ./dataset/dino/val2017
      cp -r ./DINO/annotations ./dataset/dino

      prepare pretrained
      bash scripts/download_train.sh
      cd DINO
      CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
            --pretrain_model_path './checkpoint0011_4scale_swin'	\
            --finetune_ignore label_enc.weight class_embed	\
            --output_dir logs_swin/4scale -c config/DINO/DINO_4scale_swin.py --coco_path ../dataset/dino_data \
            --options dn_scalar=100 embed_init_tgt=TRUE \
            dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
            dn_box_noise_scale=1.0 backbone_dir='./pretrained'

</details>
<details>
      <summary>FGVC-PIM</summary>

      python tools/crop.py
      cd classify
      CUDA_VISIBLE_DEVICES=0,1 python main.py --c ./configs/cfg.yaml
      
</details>

