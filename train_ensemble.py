#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 4: 多模型集成训练
训练3个不同seed的模型，概率平均集成
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
from sklearn.metrics import roc_auc_score, roc_curve
import copy

from dataset import CTDataset
from model.model_3d import MergeModel

# ============ 超参数 ============
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3  # Phase 3: 增加正则化
PATIENCE = 15
GPU_ID = 0
TARGET_SLICES = 5
IMG_SIZE = 128
NUM_SEEDS = 3
SEEDS = [42, 123, 456]
DROPOUT_RATE = 0.4  # Phase 3: 增加dropout

BASE_DIR = '/home/ltx/桌面/npj precision oncology'
NPY_DIR = os.path.join(BASE_DIR, 'preprocessed_npy')
MANIFEST_PATH = os.path.join(NPY_DIR, 'manifest.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'code', 'ensemble_training')

NUM_CLASSES = 2


def setup_logging(output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger(f'train_{seed}')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(os.path.join(output_dir, f'train_log_seed{seed}.txt'), mode='w')
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


@torch.no_grad()
def get_predictions(model, loader, device):
    """获取模型预测概率"""
    model.eval()
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        logits, _ = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)

    return np.array(all_probs)


def train_single_model(seed, train_ids, train_labels, val_ids, val_labels, device):
    """训练单个模型"""
    logger = setup_logging(OUTPUT_DIR, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info(f"=" * 60)
    logger.info(f"Training Model with Seed {seed}")
    logger.info(f"=" * 60)

    # Dataset & DataLoader
    train_ds = CTDataset(train_ids, train_labels, NPY_DIR, augment=True,
                         target_slices=TARGET_SLICES, img_size=IMG_SIZE)
    val_ds = CTDataset(val_ids, val_labels, NPY_DIR, augment=False,
                       target_slices=TARGET_SLICES, img_size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 类别加权 CrossEntropyLoss
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32)  # [负类, 正类]
    logger.info(f"Class weights: {weight.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weight.to(device))

    # 模型
    model = MergeModel(NUM_CLASSES, dropout_rate=DROPOUT_RATE).to(device)

    # 优化器 + 调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=5
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - 5, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[5]
    )

    # 训练循环
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None

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

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            logger.info(f"  -> Best model saved (val AUC={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch} (best val AUC={best_val_auc:.4f})")
                break

    # 保存最佳模型
    model_path = os.path.join(OUTPUT_DIR, f'model_seed{seed}.pth')
    torch.save({
        'seed': seed,
        'model_state_dict': best_model_state,
        'val_auc': best_val_auc,
    }, model_path)
    logger.info(f"Model saved to {model_path}")

    return model, best_model_state, best_val_auc, val_ids, val_labels


def evaluate_ensemble(models, val_ids, val_labels, device):
    """评估集成模型"""
    val_ds = CTDataset(val_ids, val_labels, NPY_DIR, augment=False,
                       target_slices=TARGET_SLICES, img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    all_probs = []
    for model, state_dict in models:
        model.load_state_dict(state_dict)
        probs = get_predictions(model, val_loader, device)
        all_probs.append(probs)

    # 概率平均集成
    ensemble_probs = np.mean(all_probs, axis=0)
    ensemble_pred = (ensemble_probs >= 0.5).astype(int)

    ensemble_auc = roc_auc_score(val_labels, ensemble_probs)
    ensemble_acc = np.mean(ensemble_pred == np.array(val_labels))

    return ensemble_auc, ensemble_acc, ensemble_probs


def main():
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载 manifest
    with open(MANIFEST_PATH, 'r') as f:
        manifest = json.load(f)

    # 筛选训练集 (cohort='train' -> 西南)
    train_ids_all, train_labels_all = [], []
    for pid, info in manifest.items():
        if info['cohort'] == 'train':
            npy_path = os.path.join(NPY_DIR, f"{pid}.npy")
            if os.path.exists(npy_path):
                train_ids_all.append(pid)
                train_labels_all.append(info['label'])

    print(f"训练集总样本: {len(train_ids_all)}, 正例: {sum(train_labels_all)}, 负例: {len(train_labels_all) - sum(train_labels_all)}")

    # 保存所有ids用于不同seed的训练
    all_data = {
        'train_ids': train_ids_all,
        'train_labels': train_labels_all
    }
    with open(os.path.join(OUTPUT_DIR, 'all_data.json'), 'w') as f:
        json.dump(all_data, f)

    # 训练多个模型
    models = []
    individual_results = []

    for seed in SEEDS:
        # 每次使用相同的划分比例但不同的随机种子
        tr_ids, val_ids, tr_labels, val_labels = train_test_split(
            train_ids_all, train_labels_all, test_size=0.15, stratify=train_labels_all, random_state=seed
        )

        model, state_dict, val_auc, _, _ = train_single_model(
            seed, tr_ids, tr_labels, val_ids, val_labels, device
        )
        models.append((model, state_dict))
        individual_results.append({'seed': seed, 'val_auc': val_auc})

    # 评估集成
    # 使用最后一个seed的划分作为验证集
    tr_ids, val_ids, tr_labels, val_labels = train_test_split(
        train_ids_all, train_labels_all, test_size=0.15, stratify=train_labels_all, random_state=SEEDS[-1]
    )

    ensemble_auc, ensemble_acc, ensemble_probs = evaluate_ensemble(models, val_ids, val_labels, device)

    print("\n" + "=" * 60)
    print("Ensemble Training Complete")
    print("=" * 60)
    print(f"Individual model results:")
    for r in individual_results:
        print(f"  Seed {r['seed']}: val AUC = {r['val_auc']:.4f}")
    print(f"\nEnsemble results (on val set):")
    print(f"  Ensemble AUC: {ensemble_auc:.4f}")
    print(f"  Ensemble Acc: {ensemble_acc:.4f}")

    # 保存集成结果
    ensemble_results = {
        'individual_results': individual_results,
        'ensemble_auc': ensemble_auc,
        'ensemble_acc': ensemble_acc,
        'seeds': SEEDS
    }
    with open(os.path.join(OUTPUT_DIR, 'ensemble_results.json'), 'w') as f:
        json.dump(ensemble_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()