import cv2
import matplotlib.pyplot as plt
import numpy as np
import util.misc as misc
import wandb

# def equalize(x):
#     x = (x - x.min()) / (x.max()-x.min()+1e-6) * 255
#     x = x.astype(np.uint8)
#     r_image, g_image, b_image = cv2.split(x)
#     r_image_eq = cv2.equalizeHist(r_image)
#     g_image_eq = cv2.equalizeHist(g_image)
#     b_image_eq = cv2.equalizeHist(b_image)

#     image_eq = cv2.merge((r_image_eq, g_image_eq, b_image_eq))
#     return image_eq


class WANDB_LOG_IMG_CONFIG:
    mean = np.zeros(3)
    std = np.ones(3)
    factor = 1.0


def equalize(x):
    x = x * WANDB_LOG_IMG_CONFIG.std.reshape(
        1, 1, 3
    ) + WANDB_LOG_IMG_CONFIG.mean.reshape(1, 1, 3)
    if x.max() > 2.0:
        x /= WANDB_LOG_IMG_CONFIG.factor
    return x


def wandb_dump_input_output(x, ys, epoch=0, texts=""):
    """
    x: H X W X C
    y: H X W X C
    """
    if misc.is_main_process():
        n_imgs = 1 + len(ys)
        x = x.numpy()
        ys = [y.numpy().astype(float) for y in ys]
        ys = [equalize(y) for y in ys]
        x = equalize(x)
        fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
        if texts:
            fig.suptitle(texts)
        axes[0].imshow(x)
        axes[0].title.set_text(f"({x.shape[0]}, {x.shape[1]})")
        for idx, y in enumerate(ys):
            axes[1 + idx].imshow(y)
            axes[1 + idx].title.set_text(f"({y.shape[0]}, {y.shape[1]})")
        wandb.log({"vis": wandb.Image(fig), "epoch": epoch})
        plt.close(fig)


def wandb_dump_images(imgs, name="vis", epoch=0):
    """
    x: H X W X C
    y: H X W X C
    """
    if misc.is_main_process():
        n_imgs = len(imgs)
        fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
        for idx, img in enumerate(imgs):
            axes[idx].imshow(img)
        wandb.log({name: wandb.Image(fig), "epoch": epoch})
        plt.close(fig)


def compare_pos_embedding(posa, posb, ns=[0]):
    """
    posa,posa: N X (L+1) X d_emb
    """
    n, l1, d = posa.shape
    _, l2, _ = posb.shape
    dim1 = int((l1 - 1) ** 0.5)
    dim2 = int((l2 - 1) ** 0.5)
    idx = [0, d // 4, d // 2, d // 4 * 3, d - 1]
    for j in ns:
        imgs = []
        for i in idx:
            a = posa[j, 1:, i].reshape(dim1, dim1).cpu().numpy()
            b = posb[j, 1:, i].reshape(dim2, dim2).cpu().numpy()
            imgs.append(a)
            imgs.append(b)
        wandb_dump_images(imgs)


def wandb_log_metadata(metadata):
    if misc.is_main_process():
        payload = {}
        if "zero_ratio" in metadata:
            zero_ratio = metadata.get("zero_ratio")
            payload.update(
                dict(
                    zero_ratio_mean=zero_ratio.mean(),
                    zero_ratio_max=zero_ratio.max(),
                    zero_ratio_min=zero_ratio.min(),
                    zero_ratio_std=zero_ratio.std(),
                )
            )

        wandb.log(payload)
