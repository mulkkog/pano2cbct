import os
import argparse
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .dataset.datasets import Pano2CBCT
from model import AttUNet2Dto3D


# -------------------------------------------------
# Adapter dataset
# base dataset:
#   ct_norm, pano_01, prob_pano, sid
# model input:
#   (B,1,400,700)
# -------------------------------------------------
class Pano2CBCT_AttUNet(Dataset):
    def __init__(self, base: Pano2CBCT, pano_in_hw: Tuple[int, int] = (200, 350)):
        self.base = base
        self.pano_in_hw = tuple(pano_in_hw)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ct_norm, pano_01, prob_pano, sid = self.base[idx]
        x = pano_01.float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        pano_in = x[0]                                  # (1,H,W)
        return pano_in, pano_01.float(), ct_norm.float(), sid


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm01_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # DDP 저장본이면 module. 제거
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(
            "Checkpoint load mismatch.\n"
            f"missing keys: {missing}\n"
            f"unexpected keys: {unexpected}"
        )

    return ckpt


def save_npy(volume_3d: np.ndarray, out_path: str):
    if volume_3d.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={volume_3d.shape}")

    vol = volume_3d.astype(np.float32, copy=False)
    np.save(out_path, vol)


def save_preview_png(
    pano_2d: np.ndarray,
    pred_3d: np.ndarray,
    out_path: str,
    sid: str,
):
    """
    pano_2d: (200, 350)
    pred_3d: (120, 200, 350)
    """
    if pano_2d.ndim != 2:
        raise ValueError(f"Expected pano_2d shape (H,W), got {pano_2d.shape}")
    if pred_3d.ndim != 3:
        raise ValueError(f"Expected pred_3d shape (Z,H,W), got {pred_3d.shape}")

    Z, H, W = pred_3d.shape
    if (Z, H, W) != (120, 200, 350):
        raise ValueError(
            f"Model output shape changed. Expected (120, 200, 350), got {(Z,H,W)}"
        )

    mip_d = pred_3d.max(axis=0)  # (200,350)
    mip_h = pred_3d.max(axis=1)  # (120,350)
    mip_w = pred_3d.max(axis=2)  # (120, 200)

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


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    pred_dir: str,
    preview_dir: str,
    save_input_pano: bool = False,
):
    model.eval()
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    pbar = tqdm(dl, desc="inference", dynamic_ncols=True)
    for pano_in, pano_01, ct_norm, sid in pbar:
        pano_in = pano_in.to(device, non_blocking=True)

        pred = model(pano_in)  # (B,240,400,700)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        pano_np = pano_01.detach().cpu().numpy().astype(np.float32)

        if pred_np.ndim != 4:
            raise RuntimeError(f"Unexpected model output shape: {pred_np.shape}")
        if pred_np.shape[1:] != (120,200,350):
            raise RuntimeError(
                f"Model output shape changed. Expected (B,240,400,700), got {pred_np.shape}"
            )

        bs = pred_np.shape[0]
        for b in range(bs):
            sid_b = sid[b] if isinstance(sid, (list, tuple)) else sid
            sid_b = str(sid_b)

            vol = pred_np[b]   # (120, 200, 350)
            pano = pano_np[b]  # (200, 350)

            npy_path = os.path.join(pred_dir, f"pred_{sid_b}.npy")
            save_npy(vol, npy_path)

            png_path = os.path.join(preview_dir, f"preview_{sid_b}.png")
            save_preview_png(
                pano_2d=pano,
                pred_3d=vol,
                out_path=png_path,
                sid=sid_b,
            )

            if save_input_pano:
                pano_png_path = os.path.join(preview_dir, f"pano_{sid_b}.png")
                plt.figure(figsize=(7, 4))
                plt.imshow(norm01_np(pano), cmap="gray")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(pano_png_path, dpi=120, bbox_inches="tight", pad_inches=0)
                plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cbct_root",
        type=str,
        default="/home/jijang/projects/PointSearch/mpr",
    )
    ap.add_argument(
        "--pano_root",
        type=str,
        default="/home/jijang/projects/PointSearch/simpx_result",
    )
    ap.add_argument("--ids_file", type=str, default="splits/test.txt")
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--ckpt", type=str, default="best.ckpt")
    ap.add_argument("--save_dir", type=str, default="inference_attunet8")

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--base_ch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pano_in_h", type=int, default=200)
    ap.add_argument("--pano_in_w", type=int, default=350)

    ap.add_argument("--save_input_pano", action="store_true")

    args = ap.parse_args()

    setup_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[info] device = {device}")
    print(f"[info] ckpt   = {args.ckpt}")
    print(f"[info] ids    = {args.ids_file}")

    # dataset
    base = Pano2CBCT(
        cbct_root=args.cbct_root,
        pano_root=args.pano_root,
        ids_file=args.ids_file,
        cache_dir=args.cache_dir,
        return_dict=False,
        expected_ct_shape=(120,200,350),
        expected_pano_shape=(200,350),
        clip_low=-500.0,
        clip_high=2500.0,
        cache_ct=True,
    )
    ds = Pano2CBCT_AttUNet(base, pano_in_hw=(args.pano_in_h, args.pano_in_w))

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )

    print(f"[info] #test subjects = {len(ds)}")

    # model
    model = AttUNet2Dto3D(
        out_depth=120,   # 고정
        base_ch=args.base_ch,
    ).to(device)

    ckpt = load_checkpoint(model, args.ckpt, device)
    print("[info] checkpoint loaded")

    if isinstance(ckpt, dict):
        if "epoch" in ckpt:
            print(f"[info] checkpoint epoch = {ckpt['epoch']}")
        if "best_val" in ckpt:
            print(f"[info] checkpoint best_val = {ckpt['best_val']}")

    pred_dir = os.path.join(args.save_dir, "pred_nii")
    preview_dir = os.path.join(args.save_dir, "preview_png")

    run_inference(
        model=model,
        dl=dl,
        device=device,
        pred_dir=pred_dir,
        preview_dir=preview_dir,
        save_input_pano=args.save_input_pano,
    )

    print(f"[done] saved NIfTI volumes to: {pred_dir}")
    print(f"[done] saved previews      to: {preview_dir}")


if __name__ == "__main__":
    main()