import math
import sys
from typing import Iterable

import torch

from pretraining.util import lr_sched
from pretraining.util import misc


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    wavelengths,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        with torch.cuda.amp.autocast(enabled=False):
            outputs, outputs_aux = model(samples, wavelengths)
            loss = criterion(outputs, targets) + 0.4 * criterion(outputs_aux, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = 0.0
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _segmentation_metrics(logits, target, num_classes, ignore_index=None):
    pred = logits.argmax(dim=1)
    target = target.long()
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if target.numel() == 0:
        zero = torch.tensor(0.0, device=logits.device)
        return zero, zero

    indices = target * num_classes + pred
    conf = torch.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    ).float()
    intersection = conf.diag()
    union = conf.sum(1) + conf.sum(0) - intersection
    valid_classes = union > 0
    miou = (intersection[valid_classes] / union[valid_classes]).mean() * 100
    acc = intersection.sum() / conf.sum().clamp_min(1.0) * 100
    return miou, acc


@torch.no_grad()
def evaluate(data_loader, model, device, wavelengths, num_classes, ignore_index=None):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output, output_aux = model(images, wavelengths)
            loss = criterion(output, target) + 0.4 * criterion(output_aux, target)

        miou, acc = _segmentation_metrics(output, target, num_classes, ignore_index)
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["miou"].update(miou.item(), n=batch_size)
        metric_logger.meters["acc"].update(acc.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(
        "* miou {miou.global_avg:.3f} acc {acc.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            miou=metric_logger.miou,
            acc=metric_logger.acc,
            losses=metric_logger.loss,
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
