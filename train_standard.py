#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标准监督学习训练脚本 — 替代 MAML
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from dataset import CTDataset
from model.model_3d import MergeModel

# ============ 超参数 (Phase 3优化) ============
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3  # 从1e-4增加到1e-3 (增加正则化)
PATIENCE = 15
GPU_ID = 0
TARGET_SLICES = 5
IMG_SIZE = 128
SEED = 42
DROPOUT_RATE = 0.4  # 从0.3增加到0.4

BASE_DIR = '/home/ltx/桌面/npj precision oncology'
NPY_DIR = os.path.join(BASE_DIR, 'preprocessed_npy')
MANIFEST_PATH = os.path.join(NPY_DIR, 'manifest.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'code', 'standard_training')

NUM_CLASSES = 2


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(os.path.join(output_dir, 'train_log.txt'), mode='w')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    acc = np.mean(np.array(all_labels) == (np.array(all_probs) >= 0.5).astype(int))
    return avg_loss, auc, acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    acc = np.mean(np.array(all_labels) == (np.array(all_probs) >= 0.5).astype(int))
    return avg_loss, auc, acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger = setup_logging(OUTPUT_DIR)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # 加载 manifest
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    # 筛选训练集 (cohort='train' -> 西南)
    train_ids, train_labels = [], []
    for pid, info in manifest.items():
        if info['cohort'] == 'train':
            npy_path = os.path.join(NPY_DIR, f"{pid}.npy")
            if os.path.exists(npy_path):
                train_ids.append(pid)
                train_labels.append(info['label'])

    logger.info(f"训练集总样本: {len(train_ids)}, 正例: {sum(train_labels)}, 负例: {len(train_labels) - sum(train_labels)}")

    # Stratified split 85/15
    tr_ids, val_ids, tr_labels, val_labels = train_test_split(
        train_ids, train_labels, test_size=0.15, stratify=train_labels, random_state=SEED
    )
    logger.info(f"Train: {len(tr_ids)}, Val: {len(val_ids)}")

    # 保存划分
    split_info = {'train_ids': tr_ids, 'val_ids': val_ids}
    with open(os.path.join(OUTPUT_DIR, 'split.json'), 'w') as f:
        json.dump(split_info, f)

    # Dataset & DataLoader
    train_ds = CTDataset(tr_ids, tr_labels, NPY_DIR, augment=True,
                         target_slices=TARGET_SLICES, img_size=IMG_SIZE)
    val_ds = CTDataset(val_ids, val_labels, NPY_DIR, augment=False,
                       target_slices=TARGET_SLICES, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 类别加权 CrossEntropyLoss (Phase 3优化)
    # 对少数类(无反应=0)加倍权重来平衡类别分布
    n_pos = sum(tr_labels)  # 反应=1
    n_neg = len(tr_labels) - n_pos  # 无反应=0
    # 类别权重: 多数类权重=1.0, 少数类权重=2.0
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32)  # [负类, 正类]
    logger.info(f"Class weights: {weight.tolist()} (0:无反应, 1:反应)")
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    # 模型 (使用优化后的dropout rate)
    model = MergeModel(NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)

    # 优化器 + 调度器 (CosineAnnealingWarmRestarts with warmup)
    # Warmup在前5个epoch完成
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 先用线性warmup在前5个epoch
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
    )
    # 然后用CosineAnnealing
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - 5, eta_min=1e-6
    )
    # 组合调度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5]
    )

    # 训练循环
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_auc, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train Loss={train_loss:.4f} AUC={train_auc:.4f} Acc={train_acc:.4f} | "
            f"Val Loss={val_loss:.4f} AUC={val_auc:.4f} Acc={val_acc:.4f} | "
            f"LR={lr:.6f}"
        )

        # Early stopping on val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
            }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
            logger.info(f"  -> Best model saved (val AUC={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch} (best val AUC={best_val_auc:.4f})")
                break

    logger.info(f"训练完成. Best val AUC={best_val_auc:.4f}")


if __name__ == '__main__':
    main()
