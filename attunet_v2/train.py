import os
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb

from dataset.datasets import Pano2CBCT
from model import AttUNet2Dto3D


def norm01_np(x):
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def save_preview_png(
    pano_2d: np.ndarray,
    pred_3d: np.ndarray,
    out_path: str,
    sid: str,
):

    if pano_2d.ndim != 2:
        raise ValueError(f"Expected pano_2d shape (H,W), got {pano_2d.shape}")
    if pred_3d.ndim != 3:
        raise ValueError(f"Expected pred_3d shape (Z,H,W), got {pred_3d.shape}")

    Z, H, W = pred_3d.shape
    if (Z, H, W) != (120, 200, 350):
        raise ValueError(
            f"Model output shape changed. Expected (120, 200, 350), got {(Z,H,W)}"
        )

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


def as_5d(pred):
    return pred.unsqueeze(1)


def mip2d(vol, axis):
    return torch.amax(vol, dim=axis)


def mip_loss(pred, gt):

    s = pred.new_tensor(0.0)

    for ax in (2, 3, 4):
        s = s + F.mse_loss(mip2d(pred, ax), mip2d(gt, ax))

    return s / 3.0


def compute_loss(pred, gt, w_l2, w_mip):

    pred = as_5d(pred)

    l2 = F.mse_loss(pred, gt)
    mip = mip_loss(pred, gt) if w_mip > 0 else pred.new_tensor(0.0)

    total = w_l2 * l2 + w_mip * mip

    return total, l2.detach(), mip.detach()


@torch.no_grad()
def run_validation(model, val_dl, device, w_l2, w_mip):

    model.eval()

    s_loss = 0
    s_l2 = 0
    s_mip = 0
    n = 0

    for pano, ct, sid in val_dl:

        pano = pano.to(device)
        ct = ct.to(device)

        pred = model(pano)

        loss, l2v, mipv = compute_loss(pred, ct, w_l2, w_mip)

        s_loss += float(loss.cpu())
        s_l2 += float(l2v.cpu())
        s_mip += float(mipv.cpu())

        n += 1

    return s_loss / n, s_l2 / n, s_mip / n


def make_preview(model, val_dl, device, epoch, save_dir):

    model.eval()

    pano, ct, sid = next(iter(val_dl))

    pano = pano.to(device)

    with torch.no_grad():
        pred = model(pano)

    pred = as_5d(pred)

    pano_np = pano[0, 0].cpu().numpy()
    pred_np = pred[0, 0].cpu().numpy()

    sid = sid[0]

    out_path = os.path.join(save_dir, f"preview_ep{epoch:03d}_{sid}.png")

    save_preview_png(
        pano_np,
        pred_np,
        out_path,
        sid,
    )

    wandb.log(
        {
            "preview": wandb.Image(out_path),
            "epoch": epoch,
        },
        step=epoch,
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cbct_root", default="data/mpr")
    ap.add_argument("--pano_root", default="data/pano")

    ap.add_argument("--train_ids_file", default="splits/train.txt")
    ap.add_argument("--val_ids_file", default="splits/test.txt")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--w_l2", type=float, default=0.5)
    ap.add_argument("--w_mip", type=float, default=1.0)

    ap.add_argument("--base_ch", type=int, default=32)

    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--save_dir", default="runs")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(
        project="pano2cbct",
        config=vars(args),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = Pano2CBCT(
        args.cbct_root,
        args.pano_root,
        args.train_ids_file,
    )

    val_ds = Pano2CBCT(
        args.cbct_root,
        args.pano_root,
        args.val_ids_file,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = AttUNet2Dto3D(
        out_depth=120,
        base_ch=args.base_ch,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device == "cuda"))

    best_val = float("inf")

    for ep in range(1, args.epochs + 1):

        model.train()

        s_loss = 0
        s_l2 = 0
        s_mip = 0

        steps_bar = tqdm(train_dl, desc=f"train {ep:03d}")

        for it, (pano, ct, sid) in enumerate(steps_bar, start=1):

            pano = pano.to(device)
            ct = ct.to(device)

            opt.zero_grad(set_to_none=True)

            if args.use_amp and device == "cuda":

                with torch.cuda.amp.autocast():

                    pred = model(pano)

                    loss, l2v, mipv = compute_loss(
                        pred, ct, args.w_l2, args.w_mip
                    )

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            else:

                pred = model(pano)

                loss, l2v, mipv = compute_loss(
                    pred, ct, args.w_l2, args.w_mip
                )

                loss.backward()
                opt.step()

            s_loss += float(loss.cpu())
            s_l2 += float(l2v.cpu())
            s_mip += float(mipv.cpu())

            steps_bar.set_postfix(
                loss=f"{s_loss/it:.4f}",
                l2=f"{s_l2/it:.4f}",
                mip=f"{s_mip/it:.4f}",
            )

        tr_loss = s_loss / len(train_dl)
        tr_l2 = s_l2 / len(train_dl)
        tr_mip = s_mip / len(train_dl)

        val_loss, val_l2, val_mip = run_validation(
            model,
            val_dl,
            device,
            args.w_l2,
            args.w_mip,
        )

        print(
            f"[train {ep:03d}] loss={tr_loss:.6f}  l2={tr_l2:.6f}  mip={tr_mip:.6f}"
        )
        print(
            f"[val   {ep:03d}] loss={val_loss:.6f}  l2={val_l2:.6f}  mip={val_mip:.6f}"
        )

        wandb.log(
            {
                "train/loss": tr_loss,
                "train/l2": tr_l2,
                "train/mip": tr_mip,
                "val/loss": val_loss,
                "val/l2": val_l2,
                "val/mip": val_mip,
                "epoch": ep,
            },
            step=ep,
        )

        if ep == 1 or ep % 10 == 0:
            make_preview(model, val_dl, device, ep, args.save_dir)

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "best.ckpt"),
            )

    wandb.finish()


if __name__ == "__main__":
    main()
