export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node=2 --master_port=25678 main_pretrain_ofa.py \
--data_path_s1 /home/zhitong/Datasets/Sentinel/ \
--data_path_s2 /home/zhitong/Datasets/Sentinel/ \
--data_path_naip /home/zhitong/Datasets/NAIP/ \
--data_path_hyper /home/zhitong/Datasets/Hyperspectral/ \
--data_path_gaufen /home/zhitong/Datasets/Gaofen/FiveBillion/ \
--output_dir checkpoints/mae_base_ofa_all_vit16_224_Ball \
--log_dir checkpoints/mae_base_ofa_all_vit16_224_Ball/log \
--model mae_vit_base_patch16 \
--norm_pix_loss \
--mask_ratio 0.75 \
--num_workers 4 \
--batch_size 96 \
--epochs 120 \
--warmup_epochs 20 \
--blr 1.0e-4 \
--weight_decay 0.05 \
--seed 42 \
--input_size 224 \
--in_chans 2 9 3 202 4 \
--split train_100k \
#--dist_url $dist_url \
#--dist_backend 'nccl' \
