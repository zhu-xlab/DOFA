import argparse
import sys
import datetime
import json
import os
import time
from dataclasses import dataclass
from typing import List
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import kornia as K
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models.layers import trunc_normal_
from torch.utils.tensorboard import SummaryWriter

import dofa_v1
from downstream_tasks.geobench_classification.engine import evaluate, train_one_epoch
from pretraining.util import misc
from pretraining.util.lars import LARS
from pretraining.util.misc import NativeScalerWithGradNormCount as NativeScaler
from pretraining.util.pos_embed import interpolate_pos_embed


@dataclass(frozen=True)
class TaskConfig:
    nb_classes: int
    band_names: List[str]
    wavelengths: List[float]
    blr: float
    checkpoint_arg: str = "finetune_ball"
    multilabel: bool = False


TASK_CONFIGS = {
    "m-forestnet": TaskConfig(
        nb_classes=12,
        band_names=[
            "04 - Red",
            "03 - Green",
            "02 - Blue",
            "05 - NIR",
            "05 - NIR",
            "05 - NIR",
            "05 - NIR",
            "06 - SWIR1",
            "07 - SWIR2",
        ],
        wavelengths=[0.66, 0.56, 0.49, 0.86, 0.86, 0.86, 0.86, 1.61, 2.2],
        checkpoint_arg="finetune_b9",
        blr=0.5,
    ),
    "m-bigearthnet": TaskConfig(
        nb_classes=43,
        band_names=[
            "04 - Red",
            "03 - Green",
            "02 - Blue",
            "05 - Vegetation Red Edge",
            "06 - Vegetation Red Edge",
            "07 - Vegetation Red Edge",
            "08 - NIR",
            "11 - SWIR",
            "12 - SWIR",
        ],
        wavelengths=[0.66, 0.56, 0.49, 0.7, 0.74, 0.78, 0.84, 1.61, 2.2],
        blr=5.0,
        multilabel=True,
    ),
    "m-brick-kiln": TaskConfig(
        nb_classes=2,
        band_names=[
            "04 - Red",
            "03 - Green",
            "02 - Blue",
            "05 - Vegetation Red Edge",
            "06 - Vegetation Red Edge",
            "07 - Vegetation Red Edge",
            "08 - NIR",
            "11 - SWIR",
            "12 - SWIR",
        ],
        wavelengths=[0.66, 0.56, 0.49, 0.7, 0.74, 0.78, 0.84, 1.61, 2.2],
        checkpoint_arg="finetune_b9",
        blr=10.0,
    ),
    "m-pv4ger": TaskConfig(
        nb_classes=2,
        band_names=["Red", "Green", "Blue"],
        wavelengths=[0.66, 0.56, 0.49],
        checkpoint_arg="finetune_b3",
        blr=1.0,
    ),
    "m-so2sat": TaskConfig(
        nb_classes=17,
        band_names=[
            "04 - Red",
            "03 - Green",
            "02 - Blue",
            "05 - Vegetation Red Edge",
            "06 - Vegetation Red Edge",
            "07 - Vegetation Red Edge",
            "08 - NIR",
            "11 - SWIR",
            "12 - SWIR",
        ],
        wavelengths=[0.66, 0.56, 0.49, 0.7, 0.74, 0.78, 0.84, 1.61, 2.2],
        checkpoint_arg="finetune_b9",
        blr=10.0,
    ),
    "m-eurosat": TaskConfig(
        nb_classes=10,
        band_names=[
            "04 - Red",
            "03 - Green",
            "02 - Blue",
            "05 - Vegetation Red Edge",
            "06 - Vegetation Red Edge",
            "07 - Vegetation Red Edge",
            "08 - NIR",
            "11 - SWIR",
            "12 - SWIR",
        ],
        wavelengths=[0.66, 0.56, 0.49, 0.7, 0.74, 0.78, 0.84, 1.61, 2.2],
        blr=10.0,
    ),
}

ORIGINAL_SCRIPT_TASKS = ["m-forestnet", "m-brick-kiln", "m-so2sat"]


class DataAugmentation(torch.nn.Module):
    def __init__(self, mean, std, split="valid", image_size=224):
        super().__init__()
        if split == "train":
            self.transform = torch.nn.Sequential(
                K.augmentation.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.Normalize(mean=mean, std=std),
            )
        else:
            self.transform = torch.nn.Sequential(
                K.augmentation.Resize(size=(image_size, image_size)),
                K.augmentation.Normalize(mean=mean, std=std),
            )

    @torch.no_grad()
    def forward(self, x):
        return self.transform(x)


class GeoBenchTransform:
    def __init__(self, task, split, band_names=None, image_size=224):
        mean, std = task.get_dataset(band_names=band_names).normalization_stats()
        self.band_names = band_names
        self.transform = DataAugmentation(mean=mean, std=std, split=split, image_size=image_size)

    def __call__(self, sample):
        array, _ = sample.pack_to_3d(
            band_names=self.band_names,
            resample=False,
            fill_value=None,
            resample_order=3,
        )
        array = torch.from_numpy(array.astype("float32")).permute(2, 0, 1)
        array = self.transform(array).squeeze(0)
        return array, torch.tensor(sample.label)


def get_args_parser():
    parser = argparse.ArgumentParser("DOFA GeoBench linear probing")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--model", default="vit_base_patch16", type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--data_path", default="/home/zhitong/Datasets/geobench/", type=str)
    parser.add_argument("--output_dir", default="outputs/geobench_linear_probe")
    parser.add_argument("--log_dir", default="outputs/geobench_linear_probe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--global_pool", action="store_true")
    parser.add_argument("--cls_token", action="store_false", dest="global_pool")
    parser.set_defaults(global_pool=True)
    parser.add_argument("--finetune", default="")
    parser.add_argument("--finetune_ball", default="checkpoints/DOFA_ViT_base_e100.pth")
    parser.add_argument("--finetune_b9", default="")
    parser.add_argument("--finetune_b3", default="")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=ORIGINAL_SCRIPT_TASKS,
        help="GeoBench classification task ids, or 'all'.",
    )
    return parser


def resolve_tasks(tasks):
    if len(tasks) == 1 and tasks[0] == "all":
        return list(TASK_CONFIGS.keys())
    unknown = sorted(set(tasks) - set(TASK_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown GeoBench task(s): {unknown}")
    return tasks


def create_dataset(task, config, image_size):
    train_transform = GeoBenchTransform(task, "train", config.band_names, image_size)
    val_transform = GeoBenchTransform(task, "valid", config.band_names, image_size)
    dataset_train = task.get_dataset(split="train", transform=train_transform, band_names=config.band_names)
    dataset_val = task.get_dataset(split="valid", transform=val_transform, band_names=config.band_names)
    dataset_test = task.get_dataset(split="test", transform=val_transform, band_names=config.band_names)
    return dataset_train, dataset_val, dataset_test


def load_pretrained(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
    state_dict = model.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]
    interpolate_pos_embed(model, checkpoint_model, num_patches=196)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    trunc_normal_(model.head.weight, std=0.01)


def make_linear_probe_model(args, config, checkpoint_path):
    model = dofa_v1.__dict__[args.model](
        num_classes=config.nb_classes,
        global_pool=args.global_pool,
    )
    if checkpoint_path and not args.eval:
        print(f"Load pre-trained checkpoint from: {checkpoint_path}")
        load_pretrained(model, checkpoint_path)

    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
        model.head,
    )
    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for _, parameter in model.head.named_parameters():
        parameter.requires_grad = True
    return model


def get_checkpoint_path(args, config):
    explicit = args.finetune
    if explicit:
        return explicit
    path = getattr(args, config.checkpoint_arg)
    return path or args.finetune_ball


def main(args):
    import geobench

    os.environ.setdefault("GEO_BENCH_DIR", args.data_path)
    selected_tasks = resolve_tasks(args.tasks)
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    max_accuracies = {}
    requested_lr = args.lr
    available_tasks = {task.dataset_name: task for task in geobench.task_iterator(benchmark_name="classification_v1.0")}

    for task_name in selected_tasks:
        task = available_tasks[task_name]
        config = TASK_CONFIGS[task_name]
        run_blr = config.blr if args.blr is None else args.blr
        checkpoint_path = get_checkpoint_path(args, config)

        print("******************************************")
        print(f"Task: {task_name}")

        output_dir = Path(args.output_dir) / task_name
        log_dir = Path(args.log_dir) / task_name
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_train, dataset_val, dataset_test = create_dataset(task, config, args.input_size)
        print(dataset_train)
        print(dataset_val)
        print(dataset_test)

        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        log_writer = None
        if global_rank == 0 and args.log_dir is not None and not args.eval:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_writer = SummaryWriter(log_dir=str(log_dir))

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        model = make_linear_probe_model(args, config, checkpoint_path)
        model.to(device)
        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model = %s" % str(model_without_ddp))
        print("number of trainable params (M): %.2f" % (n_parameters / 1.0e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        args.lr = run_blr * eff_batch_size / 256 if requested_lr is None else requested_lr
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_scaler = NativeScaler()
        criterion = torch.nn.MultiLabelSoftMarginLoss() if config.multilabel else torch.nn.CrossEntropyLoss()
        print("criterion = %s" % str(criterion))
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            test_stats = evaluate(
                data_loader_test,
                model,
                device,
                config.wavelengths,
                multilabel=config.multilabel,
                num_labels=config.nb_classes,
            )
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
            continue

        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy_val = 0.0
        max_accuracy_val_test = 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model,
                criterion,
                data_loader_train,
                optimizer,
                device,
                epoch,
                loss_scaler,
                config.wavelengths,
                max_norm=None,
                log_writer=log_writer,
                args=args,
            )
            if epoch % 10 == 9 and output_dir:
                task_args = argparse.Namespace(**vars(args))
                task_args.output_dir = str(output_dir)
                misc.save_model(
                    args=task_args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

            val_stats = evaluate(
                data_loader_val,
                model,
                device,
                config.wavelengths,
                multilabel=config.multilabel,
                num_labels=config.nb_classes,
            )
            test_stats = evaluate(
                data_loader_test,
                model,
                device,
                config.wavelengths,
                multilabel=config.multilabel,
                num_labels=config.nb_classes,
            )
            print(f"Accuracy of the network on the {len(dataset_val)} val images: {val_stats['acc1']:.1f}%")
            print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")

            if val_stats["acc1"] > max_accuracy_val:
                max_accuracy_val = val_stats["acc1"]
                max_accuracy_val_test = test_stats["acc1"]
            print(f"Max val accuracy: {max_accuracy_val:.2f}%, test accuracy: {max_accuracy_val_test:.2f}%")

            if log_writer is not None:
                log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
                log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
                log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(output_dir / "log.txt", mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        max_accuracies[task_name] = max_accuracy_val_test
        print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))
        print(f"Finished task {task_name}")
        print("******************************************")

    print("\nTest accuracies for selected tasks:")
    print(max_accuracies)


if __name__ == "__main__":
    parser = get_args_parser()
    parsed_args = parser.parse_args()
    if parsed_args.output_dir:
        Path(parsed_args.output_dir).mkdir(parents=True, exist_ok=True)
    main(parsed_args)
