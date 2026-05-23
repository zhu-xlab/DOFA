#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export GEO_BENCH_DIR=${GEO_BENCH_DIR:-/home/zhitong/Datasets/geobench/}

python -m torch.distributed.launch \
  --nproc_per_node=${NPROC_PER_NODE:-1} \
  --master_port=${MASTER_PORT:-15676} \
  downstream_tasks/geobench_segmentation/main_finetune.py \
  --data_path "$GEO_BENCH_DIR" \
  --output_dir outputs/geobench_segmentation \
  --log_dir outputs/geobench_segmentation \
  --model vit_large_patch16 \
  --num_workers 8 \
  --epochs 20 \
  --blr 0.01 \
  --warmup_epochs 3 \
  --seed 42 \
  --layer_decay 0.65 \
  --drop_path 0.1 \
  --smoothing 0 \
  --finetune_ball ../checkpoints/dofa_large/DOFA_ViT_large_e100.pth \
  --tasks m-NeonTree
