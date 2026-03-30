#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准 PyTorch Dataset，懒加载 .npy 文件
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T


class CTDataset(Dataset):
    def __init__(self, patient_ids, labels, npy_dir, augment=False,
                 target_slices=5, img_size=128):
        self.patient_ids = patient_ids
        self.labels = labels
        self.npy_dir = npy_dir
        self.augment = augment
        self.target_slices = target_slices
        self.img_size = img_size

        # 增强的数据增强策略 (Phase 3优化)
        # - RandomVerticalFlip: 新增
        # - RandomAffine: degrees=10→20, translate=(0.05, 0.05)→(0.15, 0.15)
        # - GaussianBlur: 新增 (模拟低剂量CT噪声)
        if augment:
            self.aug_transform = T.Compose([
                T.RandomVerticalFlip(p=0.5),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=20,           # 从10增加到20
                    translate=(0.15, 0.15),  # 从(0.05, 0.05)增加到(0.15, 0.15)
                    scale=(0.9, 1.1),    # 新增: 轻微缩放
                ),
                T.RandomApply([
                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.3),  # 30%概率应用高斯模糊
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = self.labels[idx]

        # 加载 .npy -> (D, H, W) uint8
        path = f"{self.npy_dir}/{pid}.npy"
        vol = np.load(path).astype(np.float32) / 255.0  # [0, 1]

        # 转 tensor (1, D, H, W)
        vol = torch.from_numpy(vol).unsqueeze(0)

        # trilinear interpolate -> (1, target_slices, img_size, img_size)
        vol = vol.unsqueeze(0)  # (1, 1, D, H, W) for F.interpolate
        vol = F.interpolate(vol, size=(self.target_slices, self.img_size, self.img_size),
                            mode='trilinear', align_corners=False)
        vol = vol.squeeze(0)  # (1, 5, 128, 128)

        # repeat channel -> (3, 5, 128, 128)
        vol = vol.repeat(3, 1, 1, 1)

        # 数据增强 (逐 slice 做 2D 增强)
        if self.aug_transform is not None:
            slices = []
            for s in range(vol.shape[1]):
                slice_2d = vol[:, s, :, :]  # (3, H, W)
                slice_2d = self.aug_transform(slice_2d)
                slices.append(slice_2d)
            vol = torch.stack(slices, dim=1)  # (3, 5, 128, 128)

        return vol, label
