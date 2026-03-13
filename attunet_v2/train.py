import os
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import wandb  # type: ignore

from dataset.datasets import Pano2CBCT
from model import AttUNet2Dto3D


class Pano2CBCT_AttUNet(Dataset):
    def __init__(self, base: Pano2CBCT, pano_in_hw=(400, 700)):
        self.base = base
        self.pano_in_hw = tuple(pano_in_hw)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ct_norm, pano_01, prob_pano, sid = self.base[idx]

        x = pano_01.float().unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=self.pano_in_hw, mode="bilinear", align_corners=False)
        pano_in = x[0]

        return pano_in, pano_01.float(), ct_norm.float(), sid


def as_5d(pred):
    return pred.unsqueeze(1)


def resize_pred_to_gt(pred, gt):
    if pred.shape[2:] == gt.shape[2:]:
        return pred
    return F.interpolate(pred, size=gt.shape[2:], mode="trilinear", align_corners=False)


def mip2d(vol, axis):
    return torch.amax(vol, dim=axis, keepdim=False)


def mip_loss(pred, gt, axes=(2, 3, 4)):
    s = pred.new_tensor(0.0)
    for ax in axes:
        s = s + F.mse_loss(mip2d(pred, ax), mip2d(gt, ax))
    return s / float(len(axes))


def compute_loss(pred, gt, w_l2, w_mip):

    pred = as_5d(pred)
    pred = resize_pred_to_gt(pred, gt)

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

    for pano_in, pano_01, ct_norm, sid in val_dl:

        pano_in = pano_in.to(device)
        ct_norm = ct_norm.to(device)

        pred = model(pano_in)

        loss, l2v, mipv = compute_loss(pred, ct_norm, w_l2, w_mip)

        s_loss += float(loss.cpu())
        s_l2 += float(l2v.cpu())
        s_mip += float(mipv.cpu())

        n += 1

    if n == 0:
        return float("inf"), float("inf"), float("inf")

    return s_loss / n, s_l2 / n, s_mip / n


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--cbct_root", type=str, default="/home/jijang/projects/PointSearch/mpr")
    ap.add_argument("--pano_root", type=str, default="/home/jijang/projects/PointSearch/simpx_result")

    ap.add_argument("--train_ids_file", type=str, default="splits/train.txt")
    ap.add_argument("--val_ids_file", type=str, default="splits/test.txt")

    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--save_dir", type=str, default="runs_attunet")
    ap.add_argument("--save_every", type=int, default=5)

    ap.add_argument("--w_l2", type=float, default=0.5)
    ap.add_argument("--w_mip", type=float, default=1.0)

    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--pano_in_h", type=int, default=200)
    ap.add_argument("--pano_in_w", type=int, default=350)

    ap.add_argument("--base_ch", type=int, default=128)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    wb_run = None
    if wandb is not None:
        wb_run = wandb.init(project="cbct2", config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_train = Pano2CBCT(
        cbct_root=args.cbct_root,
        pano_root=args.pano_root,
        ids_file=args.train_ids_file,
        cache_dir=args.cache_dir,
        return_dict=False,
        expected_ct_shape=(120, 200, 350),
        expected_pano_shape=(200, 350),
        clip_low=-500.0,
        clip_high=2500.0,
        cache_ct=True,
    )

    base_val = Pano2CBCT(
        cbct_root=args.cbct_root,
        pano_root=args.pano_root,
        ids_file=args.val_ids_file,
        cache_dir=args.cache_dir,
        return_dict=False,
        expected_ct_shape=(120, 200, 350),
        expected_pano_shape=(200, 350),
        clip_low=-500.0,
        clip_high=2500.0,
        cache_ct=True,
    )

    ds_train = Pano2CBCT_AttUNet(base_train, pano_in_hw=(args.pano_in_h, args.pano_in_w))
    ds_val = Pano2CBCT_AttUNet(base_val, pano_in_hw=(args.pano_in_h, args.pano_in_w))

    train_dl = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    val_dl = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = AttUNet2Dto3D(
        out_depth=120,
        base_ch=args.base_ch,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device == "cuda"))

    best_val = float("inf")

    for ep in range(1, args.epochs + 1):

        model.train()

        s_loss = 0
        s_l2 = 0
        s_mip = 0

        steps_bar = tqdm(
            train_dl,
            desc=f"train {ep:03d}",
            total=len(train_dl),
            leave=False,
        )

        for it, (pano_in, pano_01, ct_norm, sid) in enumerate(steps_bar, start=1):

            pano_in = pano_in.to(device)
            ct_norm = ct_norm.to(device)

            opt.zero_grad(set_to_none=True)

            if args.use_amp and device == "cuda":

                with torch.cuda.amp.autocast():

                    pred = model(pano_in)

                    loss, l2v, mipv = compute_loss(
                        pred, ct_norm, args.w_l2, args.w_mip
                    )

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            else:

                pred = model(pano_in)

                loss, l2v, mipv = compute_loss(
                    pred, ct_norm, args.w_l2, args.w_mip
                )

                loss.backward()
                opt.step()

            s_loss += float(loss.cpu())
            s_l2 += float(l2v.cpu())
            s_mip += float(mipv.cpu())

            steps_bar.set_postfix(
                {
                    "loss": f"{s_loss/it:.4f}",
                    "l2": f"{s_l2/it:.4f}",
                    "mip": f"{s_mip/it:.4f}",
                }
            )

        n = max(1, len(train_dl))

        tr_loss = s_loss / n
        tr_l2 = s_l2 / n
        tr_mip = s_mip / n

        print(
            f"[train {ep:03d}] loss={tr_loss:.6f}  l2={tr_l2:.6f}  mip={tr_mip:.6f}"
        )

        val_loss, val_l2, val_mip = run_validation(
            model,
            val_dl,
            device,
            args.w_l2,
            args.w_mip,
        )

        print(
            f"[val   {ep:03d}] loss={val_loss:.6f}  l2={val_l2:.6f}  mip={val_mip:.6f}"
        )

        sched.step(val_loss)

        if wandb is not None:

            wandb.log(
                {
                    "train/loss": tr_loss,
                    "train/l2": tr_l2,
                    "train/mip": tr_mip,
                    "val/loss": val_loss,
                    "val/l2": val_l2,
                    "val/mip": val_mip,
                    "epoch": ep,
                    "lr": opt.param_groups[0]["lr"],
                },
                step=ep,
            )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "epoch": ep,
                    "best_val": best_val,
                    "state_dict": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                },
                os.path.join(args.save_dir, "best.ckpt"),
            )

        if ep % args.save_every == 0 or ep == args.epochs:

            torch.save(
                {
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "best_val": best_val,
                },
                os.path.join(args.save_dir, f"epoch{ep:03d}.ckpt"),
            )

    print("done")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
