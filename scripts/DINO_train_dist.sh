cd DINO
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
	--pretrain_model_path './checkpoint0011_4scale_swin'	\
	--finetune_ignore label_enc.weight class_embed	\
	--output_dir logs_swin/4scale -c config/DINO/DINO_4scale_swin.py --coco_path ../dataset/dino_data \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir='./pretrained'

