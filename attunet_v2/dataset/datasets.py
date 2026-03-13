import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class Pano2CBCT(Dataset):

    def __init__(self, cbct_root, pano_root, ids_file):

        self.cbct_root = cbct_root
        self.pano_root = pano_root

        with open(ids_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.ids)

    def load_ct(self, sid):

        path = os.path.join(self.cbct_root, f"{sid}.npy")

        vol = np.load(path, mmap_mode="r")
        vol = vol.astype(np.float32).copy()

        return torch.from_numpy(vol).unsqueeze(0)

    def load_pano(self, sid):

        npy_path = os.path.join(self.pano_root, f"{sid}.npy")
        png_path = os.path.join(self.pano_root, sid, "pano_final.png")

        if os.path.exists(npy_path):

            pano = np.load(npy_path, mmap_mode="r")
            pano = pano.astype(np.float32).copy()

        else:

            pano = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
            pano = pano.astype(np.float32) / 255.0

        return torch.from_numpy(pano).unsqueeze(0)

    def load_pano_image(self, img_path):

        pano = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        pano = pano.astype(np.float32) / 255.0

        return torch.from_numpy(pano).unsqueeze(0)

    def __getitem__(self, idx):

        sid = self.ids[idx]

        pano = self.load_pano(sid)
        ct = self.load_ct(sid)

        return pano, ct, sid
