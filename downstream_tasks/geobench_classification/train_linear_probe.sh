#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export GEO_BENCH_DIR=${GEO_BENCH_DIR:-/home/zhitong/Datasets/geobench/}

python -m torch.distributed.launch \
  --nproc_per_node=${NPROC_PER_NODE:-1} \
  --master_port=${MASTER_PORT:-25627} \
  downstream_tasks/geobench_classification/main_linear_probe.py \
  --data_path "$GEO_BENCH_DIR" \
  --output_dir outputs/geobench_linear_probe \
  --log_dir outputs/geobench_linear_probe \
  --model vit_base_patch16 \
  --num_workers 4 \
  --batch_size 256 \
  --epochs 50 \
  --global_pool \
  --warmup_epochs 0 \
  --seed 42 \
  --finetune_ball checkpoints/DOFA_ViT_base_e100.pth \
  --tasks m-forestnet m-brick-kiln m-so2sat
