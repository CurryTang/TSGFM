dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="cora"
[ -z "${device}" ] && device=-1


singularity exec --nv /mnt/home/chenzh85/pytorch.sif python transductive_ssl.py \
	--device 0 \
	--dataset "cora" \
	--mask_method "random" \
    --remask_method "fixed" \
	--mask_rate 0.5 \
	--in_drop 0.2 \
	--attn_drop 0.1 \
	--num_layers 2 \
	--num_dec_layers 1 \
	--num_hidden 256 \
	--num_heads 4 \
	--num_out_heads 1 \
	--encoder "gat" \
	--decoder "gat" \
	--max_epoch 1000 \
	--max_epoch_f 300 \
	--lr 0.001 \
	--weight_decay 0.04 \
	--lr_f 0.005 \
	--weight_decay_f 1e-4 \
	--activation "prelu" \
	--loss_fn "sce" \
	--alpha_l 3 \
	--seeds 0 \
	--lam 0.5 \
	--use_cfg "/mnt/home/chenzh85/graphlang/PyGFM/MyOFA/configs/mae/cora.yaml" \
	--linear_prob \
	--scheduler
	