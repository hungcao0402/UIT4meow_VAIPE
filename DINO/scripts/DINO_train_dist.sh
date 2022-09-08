coco_path=$1
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 main.py \
	--pretrain_model_path './checkpoint0022_5scale.pth'	\
	--finetune_ignore label_enc.weight class_embed	\
	--output_dir logs/5scale -c config/DINO/DINO_5scale.py --coco_path $coco_path \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

