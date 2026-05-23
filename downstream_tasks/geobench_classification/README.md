# GeoBench Image Classification Linear Probing

This directory contains the DOFA linear probing entrypoint for GeoBench image classification. It is the cleaned-up version of `train_linear_geobench_cls_mae_all_ofa.sh`.

## Structure

- `main_linear_probe.py`: task configuration, GeoBench datasets, model loading, and train/eval orchestration.
- `engine.py`: one-epoch training and evaluation loops.
- `train_linear_probe.sh`: runnable launch script.

The code reuses the repository-level DOFA model in `dofa_v1.py` and shared training helpers from `pretraining/util`, so it does not duplicate `models_dwv.py` or copy another utility folder into the downstream task.

## Run

From the DOFA repository root:

```bash
bash downstream_tasks/geobench_classification/train_linear_probe.sh
```

By default the script runs the same tasks that the original shell script actually trained:

```text
m-forestnet m-brick-kiln m-so2sat
```

To run every configured GeoBench classification task:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=25627 \
  -m downstream_tasks.geobench_classification.main_linear_probe \
  --data_path /path/to/geobench \
  --finetune_ball checkpoints/DOFA_ViT_base_e100.pth \
  --tasks all
```

The launch script reads `GEO_BENCH_DIR`, `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, and `MASTER_PORT` from the environment when set.
