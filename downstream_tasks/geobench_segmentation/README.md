# GeoBench Segmentation Finetuning

This directory contains a cleaned-up DOFA finetuning entrypoint for GeoBench semantic segmentation. It is based on `train_finetune_geobench_seg_mae_ofa.sh`.

## Structure

- `main_finetune.py`: task configuration, GeoBench datasets, UPerNet model assembly, checkpoint loading, and training orchestration.
- `model.py`: DOFA ViT segmentation backbone that returns four intermediate feature maps for UPerNet.
- `engine.py`: training and evaluation loops with mIoU/pixel accuracy metrics.
- `train_finetune.sh`: runnable launch script.

The segmentation backbone reuses the repository-level `wave_dynamic_layer.py` instead of copying the older `models_dwv_seg.py`. Shared distributed/checkpoint utilities are reused from `pretraining/util`.

## Run

From the DOFA repository root:

```bash
bash downstream_tasks/geobench_segmentation/train_finetune.sh
```

By default this runs the task that the original shell script actually trained:

```text
m-NeonTree
```

To run another configured task:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=15676 \
  downstream_tasks/geobench_segmentation/main_finetune.py \
  --data_path /path/to/geobench \
  --finetune_ball /path/to/DOFA_ViT_large_e100.pth \
  --tasks m-NeonTree
```

The launch script reads `GEO_BENCH_DIR`, `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, and `MASTER_PORT` from the environment when set.
