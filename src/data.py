"""
PyTorch Dataset for per-subject, ordered eye-tracking scanpaths over
painting/image stimuli.
"""

import os

import numpy as np
import torch
from matio import load_from_mat
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def _iter_records(obj):
    """
    Recursively walk a (possibly nested) numpy object array - the shape
    mat-io produces for MATLAB cell arrays of structs - and yield every
    leaf record exposing 'name' and 'data' fields, regardless of nesting
    depth (mat-io tends to wrap single cells in extra (1,1) layers).
    """
    arr = np.asarray(obj)

    if arr.dtype.names and "name" in arr.dtype.names and "data" in arr.dtype.names:
        for rec in arr.reshape(-1):
            yield rec
        return

    if arr.dtype == object:
        for item in arr.reshape(-1):
            yield from _iter_records(item)
        return

    raise ValueError(f"Unexpected leaf array: dtype={arr.dtype}, shape={arr.shape}")


def _scalar_str(val):
    """Unwrap a MATLAB char-array-as-numpy-string back to a plain str."""
    arr = np.asarray(val).reshape(-1)
    return str(arr[0])


class ScanpathDataset(Dataset):
    """
    One sample = one subject's ordered scanpath over one stimulus image.

    Returns a dict:
        image:     FloatTensor (3, H, W), stimulus resized to `stim_size`
        scanpath:  FloatTensor (N, 2) [x, y], coordinates rescaled to
                   match the resized image, fixation order preserved
                   (row 0 = first fixation)
        length:    int, number of fixations N (before any padding)
        stim_name: str, stimulus identifier (the Map key)
        subj_name: str, subject/trial identifier (the 'name' field,
                   e.g. '24-53-ak.eye') - handy for subject-wise splits
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(self, root, mat_path, stim_size=(256, 256), coord_order="xy"):
        """
        root:        dataset root containing `stimuli/`
        mat_path:    path to the consolidated .mat, e.g. 'trainSet/allFixData.mat'
        stim_size:   (W, H) to resize every stimulus (and rescale coords) to
        coord_order: 'xy' if data columns are [x, y]; 'yx' if [y, x]
        """
        self.root = root
        # Keys are already relative paths like 'Action/001.jpg', matching
        # Stimuli/Action/001.jpg on disk directly - no lookup table needed.
        self.stim_dir = os.path.join(root, "Stimuli")
        self.stim_size = stim_size
        self.coord_order = coord_order

        mat = load_from_mat(mat_path)
        all_data = mat["allData"]  # dict-like MatlabContainerMap: key -> stimulus

        self.samples = []  # (img_path, stim_key, scanpath (N,2) float32, subj_name)
        missing = []
        for key in all_data.keys():
            img_path = os.path.join(self.stim_dir, key)
            if not os.path.isfile(img_path):
                missing.append(key)
                continue

            for rec in _iter_records(all_data[key]):
                subj_name = _scalar_str(rec["name"])
                fix = np.asarray(rec["data"], dtype=np.float32)
                if fix.ndim != 2 or fix.shape[1] < 2:
                    continue  # malformed/empty entry - skip rather than crash
                self.samples.append((img_path, key, fix, subj_name))

        if missing:
            print(
                f"[ScanpathDataset] warning: {len(missing)} Map key(s) had no "
                f"matching file under {self.stim_dir}, e.g. {missing[:3]}"
            )

        if not self.samples:
            raise RuntimeError(
                f"No samples matched under {self.stim_dir}. Check that "
                "Map keys (e.g. 'Action/001.jpg') line up with the actual "
                "Stimuli/<Category>/<file> layout on disk."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, stim_key, fix, subj_name = self.samples[idx]

        # --- stimulus ---
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize(self.stim_size, Image.BILINEAR)
        img_tensor = TF.to_tensor(img) * 2 - 1  # (3, H, W), values in [-1, 1]

        # --- ordered scanpath ---
        fix = fix.copy()
        if self.coord_order == "yx":
            fix = fix[:, [1, 0]]

        scale_x = self.stim_size[0] / orig_w
        scale_y = self.stim_size[1] / orig_h
        fix[:, 0] *= scale_x
        fix[:, 1] *= scale_y

        scanpath = torch.from_numpy(fix)

        return {
            "image": img_tensor,
            "scanpath": scanpath,
            "length": scanpath.shape[0],
            "stim_name": stim_key,
            "subj_name": subj_name,
        }


def collate_scanpaths(batch):
    """
    Zero-pads variable-length scanpaths to the batch max so they can be
    stacked. Also returns true lengths for masking / pack_padded_sequence.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    n_coords = batch[0]["scanpath"].shape[1]
    t_max = int(lengths.max().item())

    scanpaths = torch.zeros(len(batch), t_max, n_coords, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["scanpath"].shape[0]
        scanpaths[i, :n] = b["scanpath"]

    stim_names = [b["stim_name"] for b in batch]
    return {
        "images": images,          # (B, 3, H, W)
        "scanpaths": scanpaths,    # (B, T_max, C), zero-padded past `lengths`
        "lengths": lengths,        # (B,)
        "stim_names": stim_names,
    }

def get_dataloader(
        data_path, val_data_split_ratio,
        batch_size, num_workers, seed
        ):
    # root should contain a `Stimuli/` subfolder (e.g. Stimuli/Action/001.jpg)
    dataset = ScanpathDataset(
        root=data_path,
        mat_path=f"{data_path}/allFixData.mat",
        stim_size=(256, 256),
    )

    assert val_data_split_ratio < 1 and val_data_split_ratio > 0
    train_split_ratio = 1 - val_data_split_ratio
    data_generator = torch.Generator().manual_seed(seed)

    train_data, val_data = torch.utils.data.dataset.random_split(
        dataset, 
        [train_split_ratio, val_data_split_ratio], 
        generator=data_generator
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                        shuffle=True, collate_fn=collate_scanpaths)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers,
                        shuffle=True, collate_fn=collate_scanpaths)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # root should contain a `Stimuli/` subfolder (e.g. Stimuli/Action/001.jpg)
    dataset = ScanpathDataset(
        root="trainSet",
        mat_path="trainSet/allFixData.mat",
        stim_size=(256, 256),
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_scanpaths)

    batch = next(iter(loader))
    print("images:   ", batch["images"].shape)      # (8, 3, 256, 256)
    print("scanpaths:", batch["scanpaths"].shape)   # (8, T_max, 2 or 3)
    print("lengths:  ", batch["lengths"])
    print("stimuli:  ", batch["stim_names"])

# TODO logging.info here