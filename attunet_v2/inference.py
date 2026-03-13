import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from tqdm.auto import tqdm

from model import AttUNet2Dto3D


def norm01_np(x: np.ndarray):
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def load_checkpoint(model, ckpt_path, device):

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=True)

    return ckpt


def save_preview_png(pano_2d, pred_3d, out_path, sid):

    mip_d = pred_3d.max(axis=0)
    mip_h = pred_3d.max(axis=1)
    mip_w = pred_3d.max(axis=2)

    pano_vis = norm01_np(pano_2d)
    md_vis = norm01_np(mip_d)
    mh_vis = norm01_np(mip_h)
    mw_vis = norm01_np(mip_w)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    axes[0].imshow(pano_vis, cmap="gray")
    axes[0].set_title(f"pano ({sid})")

    axes[1].imshow(md_vis, cmap="gray")
    axes[1].set_title("pred MIP@D")

    axes[2].imshow(mh_vis, cmap="gray")
    axes[2].set_title("pred MIP@H")

    axes[3].imshow(mw_vis, cmap="gray")
    axes[3].set_title("pred MIP@W")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def load_pano(pano_path):

    pano = cv2.imread(pano_path, cv2.IMREAD_GRAYSCALE)

    if pano is None:
        raise RuntimeError(f"failed to read pano: {pano_path}")

    pano = pano.astype(np.float32) / 255.0

    if pano.shape != (200, 350):
        raise RuntimeError(f"unexpected pano size {pano.shape}")

    pano_t = torch.from_numpy(pano).unsqueeze(0).unsqueeze(0)

    return pano, pano_t


@torch.no_grad()
def run_inference(model, ids, pano_root, device, save_dir):

    pred_dir = os.path.join(save_dir, "pred_npy")
    preview_dir = os.path.join(save_dir, "preview_png")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    model.eval()

    for sid in tqdm(ids, desc="inference"):

        pano_path = os.path.join(pano_root, sid, "pano_final.png")

        pano_np, pano_t = load_pano(pano_path)

        pano_t = pano_t.to(device)

        pred = model(pano_t)

        pred_np = pred[0].cpu().numpy()

        if pred_np.shape != (120, 200, 350):
            raise RuntimeError(f"unexpected output shape {pred_np.shape}")

        np.save(
            os.path.join(pred_dir, f"{sid}.npy"),
            pred_np.astype(np.float32),
        )

        save_preview_png(
            pano_np,
            pred_np,
            os.path.join(preview_dir, f"{sid}.png"),
            sid,
        )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pano_root",
        type=str,
        default="/home/jijang/projects/PointSearch/simpx_result",
    )

    ap.add_argument("--ids_file", type=str, default="splits/test.txt")

    ap.add_argument("--ckpt", type=str, default='runs/best.ckpt')

    ap.add_argument("--save_dir", type=str, default="inference")

    ap.add_argument("--base_ch", type=int, default=32)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.ids_file) as f:
        ids = [l.strip() for l in f if l.strip()]

    print("num subjects:", len(ids))

    model = AttUNet2Dto3D(
        out_depth=120,
        base_ch=args.base_ch,
    ).to(device)

    load_checkpoint(model, args.ckpt, device)

    run_inference(
        model,
        ids,
        args.pano_root,
        device,
        args.save_dir,
    )

    print("done")


if __name__ == "__main__":
    main()
