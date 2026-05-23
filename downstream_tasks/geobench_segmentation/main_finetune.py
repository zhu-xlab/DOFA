import argparse
import datetime
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import kornia as K
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from downstream_tasks.geobench_segmentation import model as dofa_seg
from downstream_tasks.geobench_segmentation.engine import evaluate, train_one_epoch
from pretraining.util import misc
from pretraining.util.misc import NativeScalerWithGradNormCount as NativeScaler
from pretraining.util.pos_embed import interpolate_pos_embed

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass(frozen=True)
class TaskConfig:
    nb_classes: int
    band_names: List[str]
    wavelengths: Optional[List[float]]
    lr: float
    batch_size: int
    checkpoint_arg: str = "finetune_ball"
    modality: str = ""
    ignore_index: Optional[int] = None


RGB_BANDS = ["Red", "Green", "Blue"]
RGB_WAVELENGTHS = [0.66, 0.56, 0.48]
S2_9_BANDS = [
    "04 - Red",
    "03 - Green",
    "02 - Blue",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "11 - SWIR",
    "12 - SWIR",
]
S2_9_WAVELENGTHS = [0.66, 0.56, 0.49, 0.7, 0.74, 0.78, 0.84, 1.61, 2.2]

TASK_CONFIGS = {
    "m-pv4ger-seg": TaskConfig(2, RGB_BANDS, RGB_WAVELENGTHS, 0.005, 16, "finetune_b3"),
    "m-cashew-plant": TaskConfig(7, S2_9_BANDS, S2_9_WAVELENGTHS, 0.001, 12),
    "m-chesapeake": TaskConfig(7, RGB_BANDS, RGB_WAVELENGTHS, 0.005, 80, "finetune_b3"),
    "m-NeonTree": TaskConfig(2, RGB_BANDS, RGB_WAVELENGTHS, 0.005, 16, "finetune_b3", "RGB"),
    "m-nz-cattle": TaskConfig(2, RGB_BANDS, RGB_WAVELENGTHS, 0.005, 16, "finetune_b3"),
    "m-SA-crop-type": TaskConfig(10, S2_9_BANDS, S2_9_WAVELENGTHS, 0.005, 80),
    "m-seasonet": TaskConfig(34, S2_9_BANDS, S2_9_WAVELENGTHS, 0.005, 64, "finetune_b9", ignore_index=0),
}
ORIGINAL_SCRIPT_TASKS = ["m-NeonTree"]


class UperNetMAE(torch.nn.Module):
    def __init__(self, backbone, neck, decode_head, aux_head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.decode_head = decode_head
        self.aux_head = aux_head

    def forward(self, x, wave_list):
        feat = self.backbone(x, wave_list)
        feat = self.neck(feat)
        out = self.decode_head(feat)
        out = self.resize(out, size=x.shape[2:], mode="bilinear", align_corners=False)
        out_aux = self.aux_head(feat)
        out_aux = self.resize(out_aux, size=x.shape[2:], mode="bilinear", align_corners=False)
        return out, out_aux

    @staticmethod
    def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
        return F.interpolate(input, size, scale_factor, mode, align_corners)


class DataAugmentation(torch.nn.Module):
    def __init__(self, mean, std, size, split="valid"):
        super().__init__()
        self.norm = K.augmentation.Normalize(mean=mean, std=std)
        if split == "train":
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.CenterCrop(size=size, align_corners=True),
                K.augmentation.RandomRotation(degrees=90, p=0.5, align_corners=True),
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            self.transform = K.augmentation.AugmentationSequential(
                K.augmentation.CenterCrop(size=size, align_corners=True),
                data_keys=["input", "mask"],
            )

    @torch.no_grad()
    def forward(self, x, y):
        x = self.norm(x)
        return self.transform(x, y)


class GeoBenchTransform:
    def __init__(self, task, split, band_names=None, image_size=224):
        mean, std = task.get_dataset(band_names=band_names).normalization_stats()
        size = (image_size, image_size)
        if task.patch_size[0] < size[0]:
            size = task.patch_size
        self.band_names = band_names
        self.transform = DataAugmentation(mean=mean, std=std, size=size, split=split)

    def __call__(self, sample):
        array, _ = sample.pack_to_3d(
            band_names=self.band_names,
            resample=True,
            fill_value=None,
            resample_order=3,
        )
        array = torch.from_numpy(array.astype("float32")).permute(2, 0, 1)
        mask = torch.from_numpy(sample.label.data.astype("float32")).squeeze(-1)
        array, mask = self.transform(array.unsqueeze(0), mask.unsqueeze(0).unsqueeze(0))
        return array.squeeze(0), mask.squeeze(0).squeeze(0)


def get_args_parser():
    parser = argparse.ArgumentParser("DOFA GeoBench segmentation finetuning")
    parser.add_argument("--batch_size", default=None, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--model", default="vit_large_patch16", type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=0.01)
    parser.add_argument("--layer_decay", type=float, default=0.65)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--data_path", default="/home/zhitong/Datasets/geobench/", type=str)
    parser.add_argument("--output_dir", default="outputs/geobench_segmentation")
    parser.add_argument("--log_dir", default="outputs/geobench_segmentation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--finetune", default="")
    parser.add_argument("--finetune_ball", default="checkpoints/DOFA_ViT_large_e100.pth")
    parser.add_argument("--finetune_b9", default="")
    parser.add_argument("--finetune_b3", default="")
    parser.add_argument("--neontree_modality", default="RGB", choices=["RGB", "Hyperspectral"])
    parser.add_argument("--tasks", nargs="+", default=ORIGINAL_SCRIPT_TASKS, help="GeoBench segmentation task ids, or 'all'.")
    return parser


def resolve_tasks(tasks):
    if len(tasks) == 1 and tasks[0] == "all":
        return list(TASK_CONFIGS.keys())
    unknown = sorted(set(tasks) - set(TASK_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown GeoBench segmentation task(s): {unknown}")
    return tasks


def resolve_config(task_name, args):
    config = TASK_CONFIGS[task_name]
    if task_name == "m-NeonTree" and args.neontree_modality == "Hyperspectral":
        return TaskConfig(2, ["Neon"], None, 0.005, 16, "", "Hyperspectral")
    return config


def create_dataset(task, config, image_size):
    train_transform = GeoBenchTransform(task, "train", config.band_names, image_size)
    val_transform = GeoBenchTransform(task, "valid", config.band_names, image_size)
    dataset_train = task.get_dataset(split="train", transform=train_transform, band_names=config.band_names)
    dataset_val = task.get_dataset(split="valid", transform=val_transform, band_names=config.band_names)
    dataset_test = task.get_dataset(split="test", transform=val_transform, band_names=config.band_names)
    return dataset_train, dataset_val, dataset_test


def get_checkpoint_path(args, config):
    if args.finetune:
        return args.finetune
    if not config.checkpoint_arg:
        return ""
    path = getattr(args, config.checkpoint_arg)
    return path or args.finetune_ball


def load_pretrained(backbone, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_model = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = backbone.state_dict()
    for key in ["head.weight", "head.bias"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]
    interpolate_pos_embed(backbone, checkpoint_model)
    msg = backbone.load_state_dict(checkpoint_model, strict=False)
    print(msg)


def build_model(args, config, crop_size, checkpoint_path):
    from mmseg.models.decode_heads import FCNHead, UPerHead
    from mmseg.models.necks import Feature2Pyramid

    backbone = dofa_seg.__dict__[args.model](
        img_size=crop_size,
        drop_path_rate=0.0,
    )
    if checkpoint_path and not args.eval:
        print(f"Load pre-trained checkpoint from: {checkpoint_path}")
        load_pretrained(backbone, checkpoint_path)

    embed_dim = 1024 if "large" in args.model else 768
    neck = Feature2Pyramid(embed_dim=embed_dim, rescales=[4, 2, 1, 0.5])
    decoder = UPerHead(
        in_channels=[embed_dim, embed_dim, embed_dim, embed_dim],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=config.nb_classes,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    )
    aux_head = FCNHead(
        in_channels=embed_dim,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=config.nb_classes,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    )
    model = UperNetMAE(backbone, neck, decoder, aux_head)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    return model


def main(args):
    import geobench

    os.environ.setdefault("GEO_BENCH_DIR", args.data_path)
    selected_tasks = resolve_tasks(args.tasks)
    requested_lr = args.lr
    requested_batch_size = args.batch_size

    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    max_accuracies = {}
    available_tasks = {task.dataset_name: task for task in geobench.task_iterator(benchmark_name="segmentation_v1.0")}

    for task_name in selected_tasks:
        task = available_tasks[task_name]
        config = resolve_config(task_name, args)
        args.lr = config.lr if requested_lr is None else requested_lr
        args.batch_size = config.batch_size if requested_batch_size is None else requested_batch_size
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

        crop_size = task.patch_size if task.patch_size[0] < args.input_size else args.input_size
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.dist_eval:
            sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            sampler_test = torch.utils.data.DistributedSampler(dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)
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

        model = build_model(args, config, crop_size, checkpoint_path)
        model.to(device)
        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Model = %s" % str(model_without_ddp))
        print("number of trainable params (M): %.2f" % (n_parameters / 1.0e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        if requested_lr is None and args.lr is None:
            args.lr = args.blr * eff_batch_size / 256
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        param_groups = [
            {"params": model_without_ddp.neck.parameters(), "lr": args.lr},
            {"params": model_without_ddp.decode_head.parameters(), "lr": args.lr},
            {"params": model_without_ddp.aux_head.parameters(), "lr": args.lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        loss_scaler = NativeScaler()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=config.ignore_index) if config.ignore_index is not None else torch.nn.CrossEntropyLoss()
        print("criterion = %s" % str(criterion))
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        if args.eval:
            test_stats = evaluate(data_loader_test, model, device, config.wavelengths, config.nb_classes, config.ignore_index)
            print(f"mIoU of the network on the {len(dataset_test)} test images: {test_stats['miou']:.1f}%")
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
                max_norm=args.clip_grad,
                log_writer=log_writer,
                args=args,
            )
            if output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
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

            val_stats = evaluate(data_loader_val, model, device, config.wavelengths, config.nb_classes, config.ignore_index)
            test_stats = evaluate(data_loader_test, model, device, config.wavelengths, config.nb_classes, config.ignore_index)
            print(f"mIoU of the network on the {len(dataset_val)} val images: {val_stats['miou']:.1f}%")
            print(f"mIoU of the network on the {len(dataset_test)} test images: {test_stats['miou']:.1f}%")

            if val_stats["miou"] > max_accuracy_val:
                max_accuracy_val = val_stats["miou"]
                max_accuracy_val_test = test_stats["miou"]
            print(f"Max val mIoU: {max_accuracy_val:.2f}%, test mIoU: {max_accuracy_val_test:.2f}%")

            if log_writer is not None:
                log_writer.add_scalar("perf/test_miou", test_stats["miou"], epoch)
                log_writer.add_scalar("perf/test_acc", test_stats["acc"], epoch)
                log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
                "best_val_miou": max_accuracy_val,
                "test_miou_at_best_val": max_accuracy_val_test,
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

    print("\nTest mIoU under best validation for selected tasks:")
    print(max_accuracies)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
