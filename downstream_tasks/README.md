# Downstream Tasks

This folder contains task-specific training examples for using DOFA on downstream remote-sensing problems.

## GeoBench Image Classification

Linear probing code for GeoBench classification lives in:

```text
downstream_tasks/geobench_classification/
```

Run the default DOFA linear probing setup from the repository root:

```bash
bash downstream_tasks/geobench_classification/train_linear_probe.sh
```

## GeoBench Segmentation

Semantic segmentation finetuning code lives in:

```text
downstream_tasks/geobench_segmentation/
```

Run the default DOFA UPerNet finetuning setup from the repository root:

```bash
bash downstream_tasks/geobench_segmentation/train_finetune.sh
```

See each task folder README for task selection, dataset path, and checkpoint options.

## Planned Examples

- object detection
- instance segmentation
- regression
