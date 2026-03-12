# train.py
import os
import argparse
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

import wandb  # type: ignore

from dataset.datasets import Pano2CBCT
from model import AttUNet2Dto3D






# -------------------------------------------------
# Adapter dataset
# Pano2CBCT -> (ct_norm, pano_01, prob_pano, sid)
# AttUNet input expects pano_in: (1, 200, 256)
# -------------------------------------------------
class Pano2CBCT_AttUNet(Dataset):
    def __init__(self, base: Pano2CBCT, pano_in_hw=(400, 700)):
        self.base = base
        self.pano_in_hw = tuple(pano_in_hw)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        ct_norm, pano_01, prob_pano, sid = self.base[idx]
        # pano_01: (H,W) -> (1,1,H,W) -> resize -> (1,200,256)
        x = pano_01.float().unsqueeze(0).unsqueeze(0)
        x = F.interpolate(x, size=self.pano_in_hw, mode="bilinear", align_corners=False)
        pano_in = x[0]  # (1,200,256)
        return pano_in, pano_01.float(), ct_norm.float(), sid


def as_5d(pred_4d: torch.Tensor) -> torch.Tensor:
    # (B,D,H,W) -> (B,1,D,H,W)
    return pred_4d.unsqueeze(1)


def resize_pred_to_gt(pred_5d: torch.Tensor, gt_5d: torch.Tensor) -> torch.Tensor:
    if pred_5d.shape[2:] == gt_5d.shape[2:]:
        return pred_5d
    return F.interpolate(pred_5d, size=gt_5d.shape[2:], mode="trilinear", align_corners=False)


def mip2d(vol_5d: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.amax(vol_5d, dim=axis, keepdim=False)


def mip_loss(pred_5d: torch.Tensor, gt_5d: torch.Tensor, axes=(2, 3, 4)) -> torch.Tensor:
    s = pred_5d.new_tensor(0.0)
    for ax in axes:
        s = s + F.mse_loss(mip2d(pred_5d, ax), mip2d(gt_5d, ax))
    return s / float(len(axes))


def compute_loss(pred_4d: torch.Tensor, gt_5d: torch.Tensor, w_l2: float, w_mip: float):
    pred_5d = as_5d(pred_4d)
    pred_5d = resize_pred_to_gt(pred_5d, gt_5d)

    l2 = F.mse_loss(pred_5d, gt_5d)
    mip = mip_loss(pred_5d, gt_5d, (2, 3, 4)) if w_mip > 0 else pred_5d.new_tensor(0.0)
    total = w_l2 * l2 + w_mip * mip
    return total, l2.detach(), mip.detach()


def _norm01_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu()
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return np.zeros(tuple(x.shape), dtype=np.float32)
    return ((x - mn) / (mx - mn)).numpy().astype(np.float32)


def _set_coord_axes(ax, shape_hw, x_label: str, y_label: str, ticks: int = 5):
    """Show simple coordinate ticks (0..size-1) on an imshow axis.

    shape_hw: (H, W) of the displayed 2D image
    """
    H, W = int(shape_hw[0]), int(shape_hw[1])

    def _tick_positions(n, k):
        if n <= 1:
            return [0]
        k = max(2, int(k))
        if k == 2:
            return [0, n - 1]
        # include 0 and n-1, plus evenly spaced mids
        return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]

    xs = _tick_positions(W, ticks)
    ys = _tick_positions(H, ticks)

    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Keep image origin consistent with array indexing (0 at top)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)



@torch.no_grad()
def run_validation_and_save_mips(
    model,
    val_dl,
    device,
    w_l2: float,
    w_mip: float,
    out_dir: str,
    epoch: int,
    max_vis: int = 4,
):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    ep_dir = os.path.join(out_dir, f"epoch_{epoch:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    s_loss = s_l2 = s_mip = 0.0
    n_batches = 0
    vis_count = 0
    saved_preview_paths = []

    pbar = tqdm(val_dl, desc=f"val {epoch:03d}", leave=False, dynamic_ncols=True)
    for pano_in, pano_01, ct_norm, sid in pbar:
        pano_in = pano_in.to(device, non_blocking=True)   # (B,1,200,256)
        ct_norm = ct_norm.to(device, non_blocking=True)   # (B,1,Z,256,256)

        pano_01 = pano_01.to(device, non_blocking=True)     # (B,Hp,Wp)
        pred = model(pano_in)  # (B,Z,200,256)
        loss, l2v, mipv = compute_loss(pred, ct_norm, w_l2, w_mip)

        # for visualization only
        pred_5d = resize_pred_to_gt(as_5d(pred), ct_norm)
        s_loss += float(loss.detach().cpu())
        s_l2 += float(l2v.detach().cpu())
        s_mip += float(mipv.detach().cpu())
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{s_loss/n_batches:.4f}",
            "l2": f"{s_l2/n_batches:.4f}",
            "mip": f"{s_mip/n_batches:.4f}",
        })

        if vis_count >= max_vis:
            continue

        B = pano_in.shape[0]
        for b in range(B):
            if vis_count >= max_vis:
                break

            pano2d = pano_01[b]        # (Hp,Wp)
            p = pred_5d[b]             # (1,Z,H,W)
            g = ct_norm[b]             # (1,Z,H,W)

            pred_mip_d = p.amax(dim=1)[0]
            pred_mip_h = p.amax(dim=2)[0]
            pred_mip_w = p.amax(dim=3)[0]

            gt_mip_d = g.amax(dim=1)[0]
            gt_mip_h = g.amax(dim=2)[0]
            gt_mip_w = g.amax(dim=3)[0]

            pano_vis = _norm01_img(pano2d)
            pmd = _norm01_img(pred_mip_d)
            pmh = _norm01_img(pred_mip_h)
            pmw = _norm01_img(pred_mip_w)
            gmd = _norm01_img(gt_mip_d)
            gmh = _norm01_img(gt_mip_h)
            gmw = _norm01_img(gt_mip_w)

            sid_b = sid[b] if isinstance(sid, (list, tuple)) else str(sid)

            
            fig, axes = plt.subplots(2, 4, figsize=(14, 7))
            ax = axes.ravel()

            # volume (after channel): (Z,H,W) == (shape[0], shape[1], shape[2])
            Z, H, W = int(p.shape[1]), int(p.shape[2]), int(p.shape[3])

            ax[0].imshow(pano_vis, cmap="gray")
            ax[0].set_title(f"pano_gt ({sid_b}) (dim0={int(pano_vis.shape[0])}, dim1={int(pano_vis.shape[1])})")
            _set_coord_axes(ax[0], pano_vis.shape, x_label="pano dim1 (W)", y_label="pano dim0 (H)")

            ax[1].imshow(pmd, cmap="gray")
            ax[1].set_title("pred MIP@D")
            _set_coord_axes(ax[1], pmd.shape, x_label=f"vol dim2 (W={W})", y_label=f"vol dim1 (H={H})")

            ax[2].imshow(pmh, cmap="gray")
            ax[2].set_title("pred MIP@H")
            _set_coord_axes(ax[2], pmh.shape, x_label=f"vol dim2 (W={W})", y_label=f"vol dim0 (Z={Z})")

            ax[3].imshow(pmw, cmap="gray")
            ax[3].set_title("pred MIP@W")
            _set_coord_axes(ax[3], pmw.shape, x_label=f"vol dim1 (H={H})", y_label=f"vol dim0 (Z={Z})")

            ax[4].axis("off")

            ax[5].imshow(gmd, cmap="gray")
            ax[5].set_title("gt MIP@D")
            _set_coord_axes(ax[5], gmd.shape, x_label=f"vol dim2 (W={W})", y_label=f"vol dim1 (H={H})")

            ax[6].imshow(gmh, cmap="gray")
            ax[6].set_title("gt MIP@H")
            _set_coord_axes(ax[6], gmh.shape, x_label=f"vol dim2 (W={W})", y_label=f"vol dim0 (Z={Z})")

            ax[7].imshow(gmw, cmap="gray")
            ax[7].set_title("gt MIP@W")
            _set_coord_axes(ax[7], gmw.shape, x_label=f"vol dim1 (H={H})", y_label=f"vol dim0 (Z={Z})")

            fig.tight_layout()
            fig.savefig(os.path.join(ep_dir, f"{vis_count:02d}_{sid_b}.png"), dpi=120)
            plt.close(fig)

            vis_count += 1

    if n_batches == 0:
        return float("inf"), float("inf"), float("inf"), []

    return s_loss / n_batches, s_l2 / n_batches, s_mip / n_batches, saved_preview_paths


def split_indices(n: int, val_ratio: float = 0.1, seed: int = 0):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * val_ratio))) if n >= 2 else 0
    n_val = min(n_val, max(0, n - 1))

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    if len(train_idx) == 0 and len(val_idx) > 0:
        train_idx = [val_idx.pop()]

    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cbct_root", type=str, default="/home/jijang/projects/PointSearch/mpr", help="mpr root로 바뀜")
    ap.add_argument("--pano_root", type=str, default="/home/jijang/projects/PointSearch/simpx_result")
    ap.add_argument("--ids_file", type=str, default=None, help="splits/train.txt")
    ap.add_argument("--cache_dir", type=str, default=None)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_dir", type=str, default="runs_attunet8_half")
    ap.add_argument("--save_every", type=int, default=5)

    ap.add_argument("--w_l2", type=float, default=0.5)
    ap.add_argument("--w_mip", type=float, default=1.0)

    ap.add_argument("--use_amp", action="store_true")

    ap.add_argument("--pano_in_h", type=int, default=200)
    ap.add_argument("--pano_in_w", type=int, default=350)
    ap.add_argument("--base_ch", type=int, default=128)

    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ---- wandb (optional) ----
    wb_run = None
    if wandb is not None:
        wb_run = wandb.init(project="cbct2", config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device == "cuda" else "cpu"

    base = Pano2CBCT(
        cbct_root=args.cbct_root,
        pano_root=args.pano_root,
        ids_file=args.ids_file,
        cache_dir=args.cache_dir,
        return_dict=False,
        expected_ct_shape=(120, 200, 350),
        expected_pano_shape=(200, 350),
        clip_low=-500.0,
        clip_high=2500.0,
        cache_ct=True,
    )
    ds_all = Pano2CBCT_AttUNet(base, pano_in_hw=(args.pano_in_h, args.pano_in_w))

    train_idx, val_idx = split_indices(len(ds_all), val_ratio=args.val_ratio, seed=args.seed)
    ds_train = Subset(ds_all, train_idx)
    ds_val = Subset(ds_all, val_idx) if len(val_idx) > 0 else None

    train_dl = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=(len(ds_train) >= args.batch_size),
    )

    val_dl = None
    if ds_val is not None and len(ds_val) > 0:
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

    # =========================================================
    # LR Scheduler 추가 (인자 추가 없음)
    # - val_dl 있으면 val_loss 기준
    # - 없으면 train_loss 기준
    # =========================================================
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-6,
        eps=1e-12,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and device == "cuda"))

    print(f"[info] device={device} amp={bool(args.use_amp and device == 'cuda')}")
    print(f"[info] total={len(ds_all)} train={len(ds_train)} val={0 if ds_val is None else len(ds_val)}")
    print(f"[info] train batches/epoch={len(train_dl)}")

    postfix_every = 10
    best_val = float("inf")

    epochs_bar = tqdm(range(1, args.epochs + 1), desc="epochs", position=0, leave=True)

    for ep in epochs_bar:
        model.train()
        s_loss = s_l2 = s_mip = 0.0

        steps_bar = tqdm(
            train_dl,
            desc=f"train {ep:03d}",
            total=len(train_dl),
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

        for it, (pano_in, pano_01, ct_norm, sid) in enumerate(steps_bar, start=1):
            pano_in = pano_in.to(device, non_blocking=True)
            ct_norm = ct_norm.to(device, non_blocking=True)
            pano_01 = pano_01.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            if args.use_amp and device == "cuda":
                with torch.amp.autocast(device_type=device_type, enabled=True):
                    pred = model(pano_in)
                    loss, l2v, mipv = compute_loss(pred, ct_norm, args.w_l2, args.w_mip)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(pano_in)
                loss, l2v, mipv = compute_loss(pred, ct_norm, args.w_l2, args.w_mip)
                loss.backward()
                opt.step()

            s_loss += float(loss.detach().cpu())
            s_l2 += float(l2v.detach().cpu())
            s_mip += float(mipv.detach().cpu())

            if (it % postfix_every == 0) or (it == len(train_dl)):
                steps_bar.set_postfix({
                    "loss": f"{s_loss/it:.4f}",
                    "l2": f"{s_l2/it:.4f}",
                    "mip": f"{s_mip/it:.4f}",
                })

        n = max(1, len(train_dl))
        tr_loss = s_loss / n
        tr_l2 = s_l2 / n
        tr_mip = s_mip / n
        print(f"[train {ep:03d}] loss={tr_loss:.6f}  l2={tr_l2:.6f}  mip={tr_mip:.6f}")
        if wandb is not None:
            wandb.log({
                "train/loss": float(tr_loss),
                "train/l2": float(tr_l2),
                "train/mip": float(tr_mip),
                "epoch": ep,
                "lr": float(opt.param_groups[0]["lr"]),
            }, step=ep)

        if val_dl is not None:
            val_vis_dir = os.path.join(args.save_dir, "val_mip_preview")
            val_loss, val_l2, val_mip, val_preview_paths = run_validation_and_save_mips(
                model=model,
                val_dl=val_dl,
                device=device,
                w_l2=args.w_l2,
                w_mip=args.w_mip,
                out_dir=val_vis_dir,
                epoch=ep,
                max_vis=4,
            )
            print(f"[val   {ep:03d}] loss={val_loss:.6f}  l2={val_l2:.6f}  mip={val_mip:.6f}  ")

            # ---- scheduler step: val_loss 기준 ----
            sched.step(val_loss)

            # ---- wandb.log ----
            if wandb is not None:
                wandb.log({
                    "val/loss": float(val_loss),
                    "val/l2": float(val_l2),
                    "val/mip": float(val_mip),
                    "epoch": ep,
                    "lr": float(opt.param_groups[0]["lr"]),
                }, step=ep)

            if val_loss < best_val:
                best_val = val_loss
                best_path = os.path.join(args.save_dir, "best.ckpt")
                torch.save(
                    {
                        "epoch": ep,
                        "best_val": best_val,
                        "state_dict": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "args": vars(args),
                    },
                    best_path,
                )
                print(f"[save-best] {best_path} (val={best_val:.6f})")

            epochs_bar.set_postfix({
                "tr": f"{tr_loss:.4f}",
                "va": f"{val_loss:.4f}",
                "best": f"{best_val:.4f}",
            })
        else:
            # val이 없으면 train loss로 scheduler step
            sched.step(tr_loss)
            epochs_bar.set_postfix({"tr": f"{tr_loss:.4f}"})

        if (ep % args.save_every) == 0 or ep == args.epochs:
            save_path = os.path.join(args.save_dir, f"epoch{ep:03d}.ckpt")
            torch.save(
                {
                    "epoch": ep,
                    "state_dict": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "args": vars(args),
                    "best_val": best_val,
                },
                save_path,
            )
            print(f"[save] {save_path}")

    print("done.")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()