import os
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
import scipy.ndimage as ndi


def bfs_farthest(adj, start):
    n = len(adj)
    dist = -np.ones(n, dtype=np.int32)
    parent = -np.ones(n, dtype=np.int32)
    q = [int(start)]
    dist[int(start)] = 0
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    far = int(np.argmax(dist))
    return far, parent, dist


def reconstruct_path(parent, start, end):
    path = [int(end)]
    cur = int(end)
    while cur != int(start) and cur >= 0:
        cur = int(parent[cur])
        if cur < 0:
            break
        path.append(cur)
    path.reverse()
    return path


def largest_component(pts, adj):
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    best = []

    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        comp = []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best):
            best = comp

    if len(best) == n:
        return pts, adj

    keep = np.array(best, dtype=np.int32)
    pts2 = pts[keep]
    old_to_new = {int(old): i for i, old in enumerate(keep.tolist())}
    keep_set = set(keep.tolist())

    adj2 = [[] for _ in range(len(keep))]
    for old in keep.tolist():
        new = old_to_new[int(old)]
        adj2[new] = [old_to_new[int(v)] for v in adj[int(old)] if int(v) in keep_set]

    return pts2, adj2


def trace_skeleton(mask):
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("Empty skeleton mask (no foreground pixels).")

    pts = np.stack([xs.astype(np.int32), ys.astype(np.int32)], axis=1)
    idx = {(int(x), int(y)): i for i, (x, y) in enumerate(pts.tolist())}

    adj = [[] for _ in range(len(pts))]
    for i, (x, y) in enumerate(pts.tolist()):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = idx.get((x + dx, y + dy))
                if j is not None:
                    adj[i].append(j)

    pts, adj = largest_component(pts, adj)
    if len(adj) == 1:
        return pts.astype(np.float32)

    u, _, _ = bfs_farthest(adj, 0)
    v, parent, _ = bfs_farthest(adj, u)
    order = reconstruct_path(parent, u, v)
    return pts[np.array(order, dtype=np.int32)].astype(np.float32)


def arc_length_t(pts):
    if pts.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    d = np.diff(pts.astype(np.float64), axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(d, axis=1))])
    if float(s[-1]) < 1e-8:
        return np.linspace(0.0, 1.0, pts.shape[0], dtype=np.float64)
    return (s / float(s[-1])).astype(np.float64)


def make_clamped_knots(n_ctrl, degree):
    k = int(degree)
    n = int(n_ctrl)
    internal_count = n - k - 1
    if internal_count < 0:
        raise ValueError("n_ctrl must be >= degree+1")
    if internal_count == 0:
        internal = np.array([], dtype=np.float64)
    else:
        internal = np.linspace(0.0, 1.0, internal_count + 2, dtype=np.float64)[1:-1]
    return np.concatenate([np.zeros(k + 1), internal, np.ones(k + 1)]).astype(np.float64)


def bspline_basis_matrix(t, knots, degree, n_ctrl):
    A = np.empty((t.size, n_ctrl), dtype=np.float64)
    for i in range(n_ctrl):
        c = np.zeros(n_ctrl, dtype=np.float64)
        c[i] = 1.0
        A[:, i] = BSpline(knots, c, degree, extrapolate=False)(t)
    A[np.isnan(A)] = 0.0
    return A


def second_diff_matrix(n):
    if n < 3:
        return np.zeros((0, n), dtype=np.float64)
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def fit_bspline_endpoints(pts, n_out, n_ctrl=54, degree=3, lam=1e-2):
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.repeat(pts[:1].astype(np.float32), n_out, axis=0)

    t = arc_length_t(pts)
    t = np.clip(t, 0.0, 1.0)

    n_ctrl = int(max(n_ctrl, degree + 1))
    degree = int(degree)
    knots = make_clamped_knots(n_ctrl, degree)
    A = bspline_basis_matrix(t, knots, degree, n_ctrl)
    D2 = second_diff_matrix(n_ctrl)

    x0, y0 = float(pts[0, 0]), float(pts[0, 1])
    x1, y1 = float(pts[-1, 0]), float(pts[-1, 1])

    if n_ctrl <= 2:
        out = np.linspace(0.0, 1.0, n_out, dtype=np.float64)[:, None]
        return ((1.0 - out) * pts[0:1] + out * pts[-1:]).astype(np.float32)

    A_f = A[:, 1:-1]
    A_c = A[:, [0, -1]]
    c_fix_x = np.array([x0, x1], dtype=np.float64)
    c_fix_y = np.array([y0, y1], dtype=np.float64)
    D_f = D2[:, 1:-1] if D2.size else D2

    M = A_f.T @ A_f
    if D2.size and lam > 0:
        M = M + lam * (D_f.T @ D_f)
    M = M + 1e-8 * np.eye(M.shape[0], dtype=np.float64)

    bx = A_f.T @ (pts[:, 0] - A_c @ c_fix_x)
    by = A_f.T @ (pts[:, 1] - A_c @ c_fix_y)

    c_free_x = np.linalg.solve(M, bx)
    c_free_y = np.linalg.solve(M, by)

    cx = np.concatenate([[x0], c_free_x, [x1]])
    cy = np.concatenate([[y0], c_free_y, [y1]])

    tu = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
    Bu = bspline_basis_matrix(tu, knots, degree, n_ctrl)
    out_x = Bu @ cx
    out_y = Bu @ cy
    return np.stack([out_x, out_y], axis=1).astype(np.float32)


def load_centerline(path, H, W, U, transpose_image=True):
    arr = np.array(Image.open(path).convert("L"))
    if transpose_image:
        arr = arr.T
    mask = arr > 0
    pts = trace_skeleton(mask)

    ih, iw = arr.shape
    sx = (H - 1) / max(iw - 1, 1)
    sy = (W - 1) / max(ih - 1, 1)
    pts[:, 0] *= sx
    pts[:, 1] *= sy

    return fit_bspline_endpoints(pts, U, n_ctrl=54, degree=3, lam=1e-2)


def compute_normals(centerline):
    cl = np.asarray(centerline, dtype=np.float32)
    d = np.gradient(cl, axis=0)
    t = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    return n.astype(np.float32)


def adjust_normals(normals):
    n = torch.from_numpy(normals.astype(np.float32))
    N = n.shape[0]
    u = torch.linspace(0, 1, N)
    m = torch.abs(u - 0.5) > 0.30
    out = n.clone()
    tweak = torch.stack([torch.zeros_like(out[m, 0]), torch.sign(out[m, 1])], dim=1)
    out[m] = F.normalize(0.85 * out[m] + 0.15 * tweak, dim=1)
    return out.numpy().astype(np.float32)


def grid_xy(cl, nxy, H, W, U, S, depth_vox, mask_dist=None):
    cl = np.asarray(cl, np.float32)
    nxy = np.asarray(nxy, np.float32)

    x = np.arange(H, dtype=np.float32)
    y = np.arange(W, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pts = np.stack([xx.reshape(-1), yy.reshape(-1)], 1)

    tree = cKDTree(cl)
    dist, idx = tree.query(pts, k=1)

    idx_f = idx.astype(np.float32)
    c = cl[idx]
    n = nxy[idx]

    d = pts - c
    t_vox = (d * n).sum(1).astype(np.float32)
    t_min = -0.5 * float(depth_vox)
    t_max = 0.5 * float(depth_vox)
    t_vox = np.clip(t_vox, t_min, t_max)

    t_idx = (t_vox - t_min) / (t_max - t_min + 1e-6) * (S - 1)
    u_idx = np.clip(idx_f, 0.0, float(U - 1))

    gx = (2.0 * t_idx / max(S - 1, 1)) - 1.0
    gy = (2.0 * u_idx / max(U - 1, 1)) - 1.0
    g = np.stack([gx, gy], 1).reshape(H, W, 2).astype(np.float32)

    if mask_dist is not None:
        mask = dist.reshape(H, W) > float(mask_dist)
        g[mask] = 2.0
    return g


def save_debug_curve(curve_img_path, centerline, normals, out_png, transpose_image=True):
    arr = np.array(Image.open(curve_img_path).convert("L"))
    if transpose_image:
        arr = arr.T

    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap="gray")
    plt.plot(centerline[:, 1], centerline[:, 0], "r-", lw=1)

    step = max(1, len(centerline) // 40)
    idx = np.arange(0, len(centerline), step)
    p = centerline[idx]
    n = normals[idx]
    q = p + 10.0 * n
    for (r0, c0), (r1, c1) in zip(p, q):
        plt.plot([c0, c1], [r0, r1], "y-", lw=0.8)

    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def apply_bone_threshold(
    vol,
    bone_thr=None,
    bone_percentile=99.0,
    keep_largest_only=True,
    binary=False,
):
    vol = np.asarray(vol, dtype=np.float32)

    if bone_thr is None:
        bone_thr = float(np.percentile(vol, bone_percentile))

    print(f"[post] threshold = {bone_thr:.6f}")

    mask = vol >= bone_thr

    if keep_largest_only:
        labels, nlab = ndi.label(mask)
        if nlab > 0:
            sizes = ndi.sum(mask, labels, index=np.arange(1, nlab + 1))
            largest = 1 + int(np.argmax(sizes))
            mask = labels == largest
            print(f"[post] connected components = {nlab}, keeping largest = {largest}")
        else:
            print("[post] no connected component found above threshold")

    if binary:
        out = mask.astype(np.uint8)
    else:
        out = vol.copy()
        out[~mask] = 0.0
        out = out.astype(np.float32)

    return out, bone_thr


@torch.no_grad()
def bend_single_subject(
    subject_code,
    mpr_path,
    curve_path,
    out_path,
    out_h=320,
    out_w=320,
    out_z=200,
    depth_vox=80.0,
    z_bs=16,
    device="cuda",
    transpose_curve=True,
    reverse_u=False,
    mpr_flipz=False,
    ref_cbct_path=None,
    bone_thr=None,
    bone_percentile=99.0,
    keep_largest_only=True,
    binary=False,
):
    # load MPR: expected on disk as .npy with shape (S, Z, U)
    mpr_szu = np.load(mpr_path).astype(np.float32)
    mpr_szu = mpr_szu[::-1, :, :]
    mpr_szu = mpr_szu.copy()

    if mpr_szu.ndim != 3:
        raise ValueError(f"Expected 3D MPR npy, got shape={mpr_szu.shape}")

    S, Z, U = mpr_szu.shape
    if (out_z is not None) and (Z != int(out_z)):
        raise ValueError(
            f"MPR Z axis is {Z}, but requested output Z is {out_z}. "
            f"For your data these should match."
        )

    # reorder to (Z, U, S) for per-z 2D warping
    mpr = np.transpose(mpr_szu, (1, 2, 0))
    if mpr_flipz:
        mpr = mpr[::-1]
    if reverse_u:
        mpr = mpr[:, ::-1, :]

    H, W = int(out_h), int(out_w)
    Z, U, S = mpr.shape

    cl = load_centerline(curve_path, H, W, U, transpose_image=transpose_curve)
    if reverse_u:
        cl = cl[::-1].copy()

    nxy = compute_normals(cl)
    nxy = adjust_normals(nxy)

    g = grid_xy(
        cl=cl,
        nxy=nxy,
        H=H,
        W=W,
        U=U,
        S=S,
        depth_vox=float(depth_vox),
        mask_dist=0.6 * float(depth_vox),
    )
    g = torch.from_numpy(g).to(device=device, dtype=torch.float32).unsqueeze(0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    debug_png = os.path.join(os.path.dirname(out_path), f"debug_curve_{subject_code}.png")
    save_debug_curve(curve_path, cl, nxy, debug_png, transpose_image=transpose_curve)

    out = np.lib.format.open_memmap(
        out_path + ".tmp.npy",
        mode="w+",
        dtype=np.float32,
        shape=(H, W, Z),
    )

    for z0 in range(0, Z, int(z_bs)):
        z1 = min(Z, z0 + int(z_bs))
        B = z1 - z0

        sl = np.asarray(mpr[z0:z1], dtype=np.float32)
        x = torch.from_numpy(sl).to(device=device).unsqueeze(1)
        gb = g.expand(B, -1, -1, -1)

        y = F.grid_sample(
            x,
            gb,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        y = y.squeeze(1).detach().cpu().numpy().astype(np.float32)
        out[:, :, z0:z1] = np.transpose(y, (1, 2, 0))

        if (z0 // int(z_bs)) % 4 == 0:
            print(f"[bend] z {z1:4d}/{Z}  batch={B}")

    out.flush()

    # post-processing: save only bone-like voxels
    vol = np.asarray(out).copy()
    print(f"[post] value range: min={vol.min():.6f}, max={vol.max():.6f}")
    print(f"[post] percentiles: {np.percentile(vol, [50, 75, 90, 95, 99, 99.5, 99.9])}")

    vol_out, used_thr = apply_bone_threshold(
        vol=vol,
        bone_thr=bone_thr,
        bone_percentile=bone_percentile,
        keep_largest_only=keep_largest_only,
        binary=binary,
    )

    if ref_cbct_path is not None and os.path.exists(ref_cbct_path):
        ref = nib.load(ref_cbct_path)
        affine = ref.affine
        header = ref.header.copy()
        header.set_data_shape((H, W, Z))
    else:
        affine = np.eye(4, dtype=np.float32)
        header = None

    img = nib.Nifti1Image(vol_out, affine=affine, header=header)
    nib.save(img, out_path)

    try:
        os.remove(out_path + ".tmp.npy")
    except Exception:
        pass

    print(f"DONE: {out_path}")
    print(f"DEBUG: {debug_png}")
    print(f"MPR input shape on disk (S,Z,U): {mpr_szu.shape}")
    print(f"Warped with (Z,U,S): {(Z, U, S)}")
    print(f"Output CBCT shape (H,W,Z): {(H, W, Z)}")
    print(f"depth_vox used: {depth_vox}")
    print(f"bone threshold used: {used_thr}")
    print(f"binary output: {binary}")
    print(f"keep largest component only: {keep_largest_only}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_code", type=str, default="45")
    ap.add_argument(
        "--mpr_path_template",
        type=str,
        default="./inference_attunet8/pred_nii/pred_{subject_code}.npy",
    )
    ap.add_argument(
        "--curve_path",
        type=str,
        default="/home/jijang/projects/PointSearch/curve_sym.png",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./inference_coarse_refine/pred_bent",
    )
    ap.add_argument("--out_h", type=int, default=320)
    ap.add_argument("--out_w", type=int, default=320)
    ap.add_argument("--out_z", type=int, default=200)
    ap.add_argument(
        "--depth_vox",
        type=float,
        default=80.0,
        help="Thickness slab in output-grid voxels.",
    )
    ap.add_argument("--z_bs", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--no_curve_transpose",
        action="store_true",
        help="Disable arr.T used in original code if result is rotated/misaligned.",
    )
    ap.add_argument(
        "--reverse_u",
        action="store_true",
        help="Reverse U direction for both centerline and MPR if mirrored along arch.",
    )
    ap.add_argument("--mpr_flipz", action="store_true")
    ap.add_argument(
        "--ref_cbct_path",
        type=str,
        default=None,
        help="Optional reference CBCT NIfTI used only to copy affine/header.",
    )

    # bone-only save options
    ap.add_argument(
        "--bone_thr",
        type=float,
        default=None,
        help="Fixed threshold. If omitted, percentile threshold is used.",
    )
    ap.add_argument(
        "--bone_percentile",
        type=float,
        default=95.0,
        help="Percentile threshold used when --bone_thr is not given.",
    )
    ap.add_argument(
        "--keep_largest_only",
        action="store_true",
        help="Keep only the largest connected component after thresholding.",
    )
    ap.add_argument(
        "--binary",
        action="store_true",
        help="Save as binary mask (0/1) instead of intensity volume.",
    )

    args = ap.parse_args()

    device = "cpu" if args.cpu else "cuda"
    mpr_path = args.mpr_path_template.format(subject_code=args.subject_code)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"pred_bent_{args.subject_code}.nii.gz")

    bend_single_subject(
        subject_code=args.subject_code,
        mpr_path=mpr_path,
        curve_path=args.curve_path,
        out_path=out_path,
        out_h=args.out_h,
        out_w=args.out_w,
        out_z=args.out_z,
        depth_vox=args.depth_vox,
        z_bs=args.z_bs,
        device=device,
        transpose_curve=(not args.no_curve_transpose),
        reverse_u=args.reverse_u,
        mpr_flipz=args.mpr_flipz,
        ref_cbct_path=args.ref_cbct_path,
        bone_thr=args.bone_thr,
        bone_percentile=args.bone_percentile,
        keep_largest_only=args.keep_largest_only,
        binary=args.binary,
    )


if __name__ == "__main__":
    main()