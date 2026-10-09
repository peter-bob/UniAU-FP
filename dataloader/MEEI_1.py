import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class MEEI(Dataset):
    # CLIP ViT-B/16 preprocess (OpenAI CLIP)
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # Left/Right swap indices for 10-D AU vector
    # [L02,L04,L06,L15,L43,R02,R04,R06,R15,R43] -> swapped
    SWAP_IDX = torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=torch.long)

    def __init__(self, root, split_txt_root, train=True, cache_labels=True, flip_p=0.5):
        self.root = os.path.expanduser(root)
        self.split_txt_root = os.path.expanduser(split_txt_root)
        self.train = train
        self.cache_labels = cache_labels
        self.flip_p = float(flip_p) if train else 0.0

        self.image_root = os.path.join(self.root, 'Image')
        self.label_root = os.path.join(self.root, 'Label')

        split_file = 'MEEI_1_train.txt' if train else 'MEEI_1_test.txt'
        split_path = os.path.join(self.split_txt_root, split_file)
        self.folder_list = self._read_txt(split_path)

        # frames used (5 frames per folder)
        self.frames = ["01", "08", "16", "24", "32"]
        self.frame_ids = [1, 8, 16, 24, 32]
        self.frameid_to_pos = {fid: i for i, fid in enumerate(self.frame_ids)}
        self.framename_to_pos = {fn: i for i, fn in enumerate(self.frames)}

        # AU columns
        self.au_columns = [
            "Left_AU02", "Left_AU04", "Left_AU06", "Left_AU15", "Left_AU43",
            "Right_AU02", "Right_AU04", "Right_AU06", "Right_AU15", "Right_AU43"
        ]

        self.samples = []
        for folder in self.folder_list:
            for f, fid in zip(self.frames, self.frame_ids):
                self.samples.append((folder, f, fid))

        # label cache: folder -> np.ndarray shape (5, num_aus)
        self._label_cache = {}

        # CLIP-friendly transforms: avoid stretching (Resize + CenterCrop)
        # (train/test keep same geometric steps; flip handled manually to swap labels)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD),
        ])

    def _read_txt(self, file):
        with open(file, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def _read_folder_xlsx(self, folder: str) -> np.ndarray:
        label_path = os.path.join(self.label_root, f'{folder}_au_results.xlsx')
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        df = pd.read_excel(label_path)

        # Prefer using 'Filename' as index if exists
        if 'Filename' in df.columns:
            df = df.set_index('Filename')

        # Try to locate rows for the 5 frames robustly
        row_keys_candidates = []

        # 1) numeric frame ids (1,8,16,24,32)
        row_keys_candidates.append(self.frame_ids)
        row_keys_candidates.append([str(x) for x in self.frame_ids])

        # 2) frame names ("01","08",...) and possible filename formats
        row_keys_candidates.append(self.frames)
        row_keys_candidates.append([f"{x}.jpg" for x in self.frames])
        row_keys_candidates.append([f"{x}.png" for x in self.frames])

        # find a usable key list
        selected = None
        for keys in row_keys_candidates:
            ok = True
            for k in keys:
                if k not in df.index:
                    ok = False
                    break
            if ok:
                selected = keys
                break

        if selected is None:
            sample_idx = list(df.index[:10])
            raise KeyError(
                f"Cannot match frame keys in xlsx index for folder={folder}. "
                f"Expected one of: {self.frame_ids} / {self.frames} / '01.jpg'... "
                f"but got index sample: {sample_idx}"
            )

        arr = df.loc[selected, self.au_columns].to_numpy(dtype=np.float32)

        # Ensure it is (5, num_aus)
        if arr.shape[0] != len(self.frames):
            raise ValueError(f"Label row mismatch for folder={folder}: got {arr.shape}")

        return arr

    def _get_folder_labels(self, folder: str) -> np.ndarray:
        if (not self.cache_labels) or (folder not in self._label_cache):
            arr = self._read_folder_xlsx(folder)
            if self.cache_labels:
                self._label_cache[folder] = arr
            return arr
        return self._label_cache[folder]

    def __getitem__(self, index):
        folder, frame_name, frame_id = self.samples[index]

        img_path = os.path.join(self.image_root, folder, f'{frame_name}.jpg')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        with Image.open(img_path) as im:
            image = im.convert('RGB')

        flipped = False
        if self.train and self.flip_p > 0 and random.random() < self.flip_p:
            image = ImageOps.mirror(image)
            flipped = True

        image = self.transform(image)

        labels_5x = self._get_folder_labels(folder)  # (5, num_aus)
        pos = self.frameid_to_pos.get(frame_id, None)
        if pos is None:
            pos = self.framename_to_pos.get(frame_name, None)
        if pos is None:
            raise KeyError(f"Unknown frame mapping: frame_id={frame_id}, frame_name={frame_name}")

        au = torch.from_numpy(labels_5x[pos])  # (num_aus,)
        if flipped:
            au = au[self.SWAP_IDX]

        return image, au
