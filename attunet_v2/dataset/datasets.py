# dataset/datasets.py
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image


# -------------------------
# NIfTI loader
# -------------------------
def _load_nifti(path: str) -> np.ndarray:
    try:
        import nibabel as nib  # type: ignore
        vol = nib.load(path).get_fdata(dtype=np.float32)
        return vol.astype(np.float32)
    except Exception as e_nib:
        try:
            import SimpleITK as sitk  # type: ignore
            img = sitk.ReadImage(path)
            vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z,Y,X)
            return vol
        except Exception as e_sitk:
            raise RuntimeError(
                f"Failed to load NIfTI: {path}\n"
                f"- nibabel error: {repr(e_nib)}\n"
                f"- SimpleITK error: {repr(e_sitk)}\n"
                "Install one: `pip install nibabel` (recommended) or `pip install SimpleITK`."
            )


# -------------------------
# Helpers
# -------------------------
def negative_likelihood(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    std = std.clamp_min(eps)
    return torch.log(std) + (image - mean) ** 2 / (2.0 * (std ** 2))


def _strip_ext(fname: str) -> str:
    if fname.lower().endswith(".nii.gz"):
        return fname[:-7]
    return os.path.splitext(fname)[0]


def _list_by_id(root: str, exts: Tuple[str, ...]) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for fn in sorted(os.listdir(root)):
        if any(fn.lower().endswith(e) for e in exts):
            sid = _strip_ext(fn)
            d[sid] = os.path.join(root, fn)
    return d


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _read_ids_file(ids_file: str) -> set:
    with open(ids_file, "r") as f:
        ids = set([line.strip() for line in f if line.strip()])
    return ids



# -------------------------
# Dataset
# -------------------------
class Pano2CBCT(Dataset):
    """
    Returns tuple by default:
      ct_norm, pano_01, prob_pano, sid

    ct_norm: (1,Z,H,W) float32 in [0,1]
    pano_01: (400,700) float32 in [0,1]
    """

    def __init__(
        self,
        cbct_root: str,
        pano_root: str,
        mean_pano_path: Optional[str] = None,
        std_pano_path: Optional[str] = None,
        *,
        # split
        ids_file: Optional[str] = None,
        expected_ct_shape: Tuple[int, int, int] = (120, 200, 350),  # (H,W,Z)
        expected_pano_shape: Tuple[int, int] = (200, 350),          # (H,W)

        clip_low: float = -500.0,
        clip_high: float = 2500.0,

        # caching
        cache_dir: Optional[str] = None,
        cache_ct: bool = True,
        cache_ct_dtype: str = "float16",   # "float16" or "float32"
        use_only_cache: bool = False,      # if True: cache miss => error

        # output format
        return_dict: bool = False,

        # debug
        debug_cache: bool = False,
    ):
        super().__init__()
        self.expected_ct_shape = tuple(expected_ct_shape)
        self.expected_pano_shape = tuple(expected_pano_shape)

        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)


        self.cache_dir = cache_dir
        self.cache_ct = bool(cache_ct) and (cache_dir is not None)
        self.cache_ct_dtype = str(cache_ct_dtype)
        self.use_only_cache = bool(use_only_cache)

        self.return_dict = bool(return_dict)
        self.debug_cache = bool(debug_cache)

        # list files
        # cbct: {cbct_root}/mpr_sigma_{subject_code}.npy
        # pano: {pano_root}/{subject_code}/pano_final.png

        if ids_file is not None:
            candidate_ids = sorted(_read_ids_file(ids_file))
        else:
            # cbct_root를 기준으로 사용 가능한 id를 추정
            candidate_ids = []
            for fn in sorted(os.listdir(cbct_root)):
                if fn.startswith("mpr_sigma_") and fn.lower().endswith(".npy"):
                    sid = fn[len("mpr_sigma_"):-4]
                    candidate_ids.append(sid)

        cbct: Dict[str, str] = {}
        pano: Dict[str, str] = {}

        for sid in candidate_ids:
            cbct_path = os.path.join(cbct_root, f"mpr_sigma_{sid}.npy")
            pano_path = os.path.join(pano_root, sid, "pano_final.png")

            if os.path.exists(cbct_path) and os.path.exists(pano_path):
                cbct[sid] = cbct_path
                pano[sid] = pano_path

        common = sorted(set(cbct.keys()) & set(pano.keys()))

        if len(common) == 0:
            raise RuntimeError(
                "No matched ids across cbct and pano after applying ids_file.\n"
                f"Expected cbct path: {cbct_root}/mpr_sigma_{{subject_code}}.npy\n"
                f"Expected pano path: {pano_root}/{{subject_code}}/pano_final.png\n"
                f"cbct_root={cbct_root}\n"
                f"pano_root={pano_root}\n"
                f"ids_file={ids_file}"
            )

        self.ids = common
        self.cbct_paths = {k: cbct[k] for k in common}
        self.pano_paths = {k: pano[k] for k in common}

        self.mean_pano = torch.from_numpy(np.load(mean_pano_path).astype(np.float32)) if mean_pano_path else None
        self.std_pano = torch.from_numpy(np.load(std_pano_path).astype(np.float32)) if std_pano_path else None

        # cache dirs
        self.ct_cache_dir = None
    def __len__(self) -> int:
        return len(self.ids)

    # -------------------------
    # pano load
    # -------------------------
    def _load_pano_png_01(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        # (H,W) = (400,700)으로 강제 리사이즈
        target_h, target_w = int(self.expected_pano_shape[0]), int(self.expected_pano_shape[1])
        if img.size != (target_w, target_h):  # PIL size는 (W,H)
            img = img.resize((target_w, target_h), resample=Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0
        # arr = arr[::-1, :]
        # arr = arr.copy()

        if arr.shape != self.expected_pano_shape:
            raise ValueError(f"Unexpected pano shape {arr.shape} for {path}, expected {self.expected_pano_shape}")

        return torch.from_numpy(arr)

    # -------------------------
    # CT shape standardization
    # -------------------------
    def _standardize_ct_to_hwc(self, vol: np.ndarray) -> np.ndarray:
        vol = vol.astype(np.float32)
        vol = np.transpose(vol, (2, 0, 1))
        vol = vol[:, ::-1, :]
        return vol

    # -------------------------
    # CT downsample + normalize
    # -------------------------
    def _downsample_and_normalize_ct(self, vol_hwc: np.ndarray) -> torch.Tensor:
        vol = np.clip(vol_hwc, self.clip_low, self.clip_high)
        vol01 = (vol - self.clip_low) / (self.clip_high - self.clip_low + 1e-12)
        ct = torch.from_numpy(vol01).unsqueeze(0).float()  # (1,Z,H,W)
        x = ct.unsqueeze(0)  # (1,1,Z,H,W)
        return x[0]  # (1,z_out,hw_out,hw_out)

    # -------------------------
    # CT cache I/O
    # -------------------------
    def _ct_cache_path(self, sid: str) -> Optional[str]:
        if not self.cache_ct or self.ct_cache_dir is None:
            return None
        return os.path.join(self.ct_cache_dir, f"{sid}.npy")

    def _load_ct_from_cache(self, sid: str) -> Optional[torch.Tensor]:
        path = self._ct_cache_path(sid)
        if path is None or (not os.path.exists(path)):
            return None
        arr = np.load(path, mmap_mode="r").copy()
        ct = torch.from_numpy(arr).unsqueeze(0)  # (1,Z,H,W)
        return ct

    def _save_ct_to_cache(self, sid: str, ct_norm: torch.Tensor):
        path = self._ct_cache_path(sid)
        if path is None or os.path.exists(path):
            return

        arr = ct_norm[0].detach().cpu().numpy()  # (Z,H,W)
        arr = arr.astype(np.float16 if self.cache_ct_dtype == "float16" else np.float32)

        tmp = path + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)
 # __getitem__
    # -------------------------
    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        sid = self.ids[index]

        # CT: cache first
        ct_norm = self._load_ct_from_cache(sid)
        cache_path = self._ct_cache_path(sid)

        if self.debug_cache and index < 3:
            print("[CT CACHE]", sid, cache_path, os.path.exists(cache_path) if cache_path else None)

        if ct_norm is None:
            if self.use_only_cache:
                raise RuntimeError(f"Cache miss for sid={sid}. Expected cache at: {cache_path}")

            # vol = _load_nifti(self.cbct_paths[sid])
            vol = np.load(self.cbct_paths[sid])
            vol_hwc = self._standardize_ct_to_hwc(vol)
            ct_norm = self._downsample_and_normalize_ct(vol_hwc)
            if self.cache_ct:
                self._save_ct_to_cache(sid, ct_norm)

        ct_norm = ct_norm.float()

        pano_01 = self._load_pano_png_01(self.pano_paths[sid]).float()

        if self.mean_pano is not None and self.std_pano is not None:
            if self.mean_pano.shape != pano_01.shape or self.std_pano.shape != pano_01.shape:
                raise ValueError(
                    f"mean/std pano shape mismatch: mean={tuple(self.mean_pano.shape)} "
                    f"std={tuple(self.std_pano.shape)} pano={tuple(pano_01.shape)}"
                )
            prob_pano = negative_likelihood(pano_01, self.mean_pano, self.std_pano).float()
        else:
            prob_pano = torch.zeros_like(pano_01)

        if self.return_dict:
            return {
                "ct_norm": ct_norm,
                "pano_01": pano_01,
                "prob_pano": prob_pano,
                "id": sid,
            }

        return ct_norm, pano_01, prob_pano, sid
