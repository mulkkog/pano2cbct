import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# input
cbct_root = "/home/jijang/projects/PointSearch/mpr"
pano_root = "/home/jijang/projects/PointSearch/simpx_result"

ids_files = [
    "splits/train.txt",
    "splits/test.txt",
]

# output
out_mpr = "data/mpr"
out_pano = "data/pano"

os.makedirs(out_mpr, exist_ok=True)
os.makedirs(out_pano, exist_ok=True)

clip_low = -500.0
clip_high = 2500.0
scale = 1.0 / (clip_high - clip_low)

# ---- ids 합치기 ----
ids = set()

for fpath in ids_files:
    with open(fpath) as f:
        for l in f:
            l = l.strip()
            if l:
                ids.add(l)

ids = sorted(ids)

print("total ids:", len(ids))

skip_count = 0

for sid in tqdm(ids):

    try:

        ct_path = os.path.join(cbct_root, f"mpr_sigma_{sid}.npy")
        pano_path = os.path.join(pano_root, sid, "pano_final.png")

        out_ct = os.path.join(out_mpr, f"{sid}.npy")
        out_p = os.path.join(out_pano, f"{sid}.npy")

        # ---- file 존재 확인 ----
        if not os.path.exists(ct_path):
            print("missing ct:", sid)
            skip_count += 1
            continue

        if not os.path.exists(pano_path):
            print("missing pano:", sid)
            skip_count += 1
            continue

        # ---- CBCT ----
        if not os.path.exists(out_ct):

            vol = np.load(ct_path)

            vol = np.transpose(vol, (2, 0, 1))
            vol = vol[:, ::-1, :]

            vol = np.clip(vol, clip_low, clip_high)
            vol = (vol - clip_low) * scale

            vol = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()

            vol = F.interpolate(
                vol,
                size=(120, 200, 350),
                mode="trilinear",
                align_corners=False,
            )

            vol = vol.squeeze().numpy().astype(np.float32)

            np.save(out_ct, vol)

        # ---- pano ----
        if not os.path.exists(out_p):

            pano = cv2.imread(pano_path, cv2.IMREAD_GRAYSCALE)

            if pano is None:
                print("corrupted pano:", sid)
                skip_count += 1
                continue

            pano = pano.astype(np.float32) / 255.0

            np.save(out_p, pano)

    except Exception as e:

        print("error on", sid, ":", e)
        skip_count += 1
        continue

print("preprocess done")
print("skipped:", skip_count)