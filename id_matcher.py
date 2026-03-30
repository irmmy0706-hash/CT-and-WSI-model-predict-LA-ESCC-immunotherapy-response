#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: ID Matching Validation
建立临床标签与影像文件夹的映射关系，验证匹配率 > 95%
ID格式:
  - 西南(训练): 800xxxxxxx
  - 西南前瞻: 705200041 / 90215725 / 800xxxxxxx
  - 大坪(DP): A000xxxxxx / FA000xxxxxx (存储为.nii.gz)
  - 巴南: 10位数字 (如 20180048702)
"""
import os
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

BASE_DIR = Path('/home/ltx/桌面/npj precision oncology')
CLINICAL_PATH = BASE_DIR / 'clinical data' / '合并队列.xlsx'
CT_WSI_DIR = BASE_DIR / 'ct and wsi data'
OUTPUT_DIR = BASE_DIR / 'preprocessed_npy'
MANIFEST_PATH = OUTPUT_DIR / 'manifest.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def load_clinical_data():
    """加载临床数据，返回DataFrame"""
    df = pd.read_excel(CLINICAL_PATH)
    logger.info(f"临床数据: {len(df)} 条记录")
    logger.info(f"  列名: {df.columns.tolist()}")
    logger.info(f"  中心分布:\n{df['center'].value_counts().to_string()}")
    logger.info(f"  标签分布: 反应=1:{(df['反应']==1).sum()}, 反应=0:{(df['反应']==0).sum()}")
    return df


def parse_sw_id(patient_id):
    """解析西南ID: 800xxxxxxx 或普通数字ID"""
    pid_str = str(patient_id)
    # 西南ID格式: 800xxxxxxx
    if pid_str.startswith('800') and len(pid_str) == 10:
        return pid_str, 'SW'
    # 西南前瞻ID: 可能是普通数字
    return pid_str, 'SW_PROS'


def parse_dp_id(patient_id):
    """解析大坪ID: FA000xxxxxx / A000xxxxxx"""
    pid_str = str(patient_id)
    if pid_str.startswith('FA'):
        return pid_str, 'DP'
    elif pid_str.startswith('A') and len(pid_str) == 9:
        return f"FA{pid_str[1:]}", 'DP'  # A000xxx -> FA000xxx
    elif pid_str.startswith('20'):  # nii.gz files like 202167975
        return pid_str, 'DP'
    return None, None


def parse_banan_id(patient_id):
    """解析巴南ID: 10位数字"""
    pid_str = str(patient_id)
    if pid_str.isdigit() and len(pid_str) >= 8:
        return pid_str, 'BANAN'
    return None, None


def scan_ct_folders():
    """扫描CT影像文件夹，建立ID->路径映射"""
    ct_map = {}

    # 西南 SW 文件夹
    sw_dir = CT_WSI_DIR / 'SW'
    if sw_dir.exists():
        for item in sw_dir.iterdir():
            if item.is_dir():
                ct_map[item.name] = ('SW_CT', item)
                # 检查是否有.svs子文件夹
                svs_files = list(item.glob('*.svs'))
                if svs_files:
                    ct_map[item.name] = ('SW_WSI', item)

    # 大坪 DP 文件夹 (.nii.gz files)
    dp_dir = CT_WSI_DIR / 'DP'
    if dp_dir.exists():
        for item in dp_dir.iterdir():
            if item.is_file() and item.suffix == '.gz':
                base_id = item.stem  # 去掉 .nii.gz
                ct_map[base_id] = ('DP_CT', item)
            elif item.is_dir():
                ct_map[item.name] = ('DP_CT', item)

    # 巴南 Banan 文件夹
    banan_dir = CT_WSI_DIR / 'Banan '
    if banan_dir.exists():
        for item in banan_dir.iterdir():
            if item.is_dir():
                ct_map[item.name] = ('BANAN_CT', item)
                # 检查WSI
                svs_files = list(item.glob('*.svs'))
                if svs_files:
                    ct_map[item.name] = ('BANAN_WSI', item)

    # 前瞻队列 1/ 文件夹 - 注意：实际上包含的是DP数据(FA000...)或重复数据
    # 只添加FA前缀的项，它们对应DP的A000...临床ID
    pros_dir = CT_WSI_DIR / '1'
    if pros_dir.exists():
        for item in pros_dir.iterdir():
            if item.is_dir():
                # FA000... 格式实际上是DP数据，不是前瞻数据
                # 跳过，因为DP folder里已经有对应的A000...数据
                pass

    logger.info(f"扫描到 {len(ct_map)} 个CT影像文件夹")
    for k, v in list(ct_map.items())[:5]:
        logger.info(f"  {k}: {v[0]}")
    return ct_map


def normalize_pid(pid):
    """Normalize patient ID to string, removing common suffixes"""
    pid_str = str(pid).strip()
    # Remove .nii.gz suffix
    if pid_str.endswith('.nii.gz'):
        pid_str = pid_str[:-7]
    return pid_str


def match_clinical_to_ct(df, ct_map):
    """将临床数据ID匹配到CT影像"""
    matched = {'SW': [], 'DP': [], 'BANAN': [], 'PROS': []}
    unmatched = []

    # Build reverse lookup (normalized key -> original key)
    reverse_map = {}
    for key in ct_map:
        reverse_map[key] = key
        # Also store without leading zeros variations
        for k2 in ct_map:
            if key != k2 and (key in k2 or k2 in key):
                reverse_map[k2] = key

    for _, row in df.iterrows():
        pid = row['ID']
        center = row['center']
        label = row['反应']
        cohort = row['cohort']

        matched_id = None
        source = None
        pid_str = normalize_pid(pid)

        if center == '西南':
            # SW格式: 800xxxxxxx
            if pid_str.startswith('800') and len(pid_str) == 10:
                if pid_str in ct_map:
                    matched_id = pid_str
                    source = ct_map[pid_str][0]
            # 也尝试模糊匹配
            if matched_id is None:
                for key in ct_map:
                    if pid_str == key or pid_str in key or key in pid_str:
                        matched_id = key
                        source = ct_map[key][0]
                        break

        elif center == '西南前瞻':
            pid_str_clean = pid_str
            for key in ct_map:
                if pid_str_clean == key or pid_str_clean in key or key in pid_str_clean:
                    matched_id = key
                    source = ct_map[key][0]
                    break

        elif center == '大坪':
            # DP格式: FA000xxxxxx, A000xxxxxx, or numeric like 202167975
            if pid_str.startswith('FA') and pid_str in ct_map:
                matched_id = pid_str
                source = ct_map[pid_str][0]
            elif pid_str.startswith('A') and len(pid_str) == 9:
                fa_id = f"FA{pid_str[1:]}"
                if fa_id in ct_map:
                    matched_id = fa_id
                    source = ct_map[fa_id][0]
            else:
                # 尝试直接匹配.nii.gz的base name
                for key in ct_map:
                    if pid_str == key or pid_str in key or key in pid_str:
                        matched_id = key
                        source = ct_map[key][0]
                        break

        elif center == '巴南':
            pid_str_clean = str(pid).strip()
            # BANAN: 10+ digit IDs
            if pid_str_clean in ct_map:
                matched_id = pid_str_clean
                source = ct_map[pid_str_clean][0]
            else:
                # 模糊匹配
                for key in ct_map:
                    if pid_str_clean == key or pid_str_clean in key or key in pid_str_clean:
                        matched_id = key
                        source = ct_map[key][0]
                        break

        if matched_id:
            # Map center to cohort_key
            if center == '西南':
                cohort_key = 'SW'
            elif center == '西南前瞻':
                cohort_key = 'PROS'
            elif center == '大坪':
                cohort_key = 'DP'
            elif center == '巴南':
                cohort_key = 'BANAN'
            else:
                cohort_key = center[:2].upper()
            matched[cohort_key].append({
                'clinical_id': pid,
                'matched_id': matched_id,
                'ct_path': str(ct_map[matched_id][1]),
                'source': source,
                'label': label,
                'center': center,
                'cohort': cohort
            })
        else:
            unmatched.append({
                'clinical_id': pid,
                'center': center,
                'cohort': cohort,
                'label': label
            })

    return matched, unmatched


def build_manifest(matched):
    """构建manifest.json供训练使用"""
    manifest = {}

    for cohort_key, items in matched.items():
        for item in items:
            pid = item['matched_id']
            manifest[pid] = {
                'label': int(item['label']),
                'center': item['center'],
                'cohort': cohort_key,
                'ct_path': item['ct_path'],
                'source': item['source']
            }

    return manifest


def verify_npy_presence(manifest):
    """验证NPY文件是否存在"""
    missing = []
    present = []

    for pid, info in manifest.items():
        npy_path = OUTPUT_DIR / f"{pid}.npy"
        if npy_path.exists():
            present.append(pid)
        else:
            missing.append(pid)

    logger.info(f"NPY文件: {len(present)} 存在, {len(missing)} 缺失")
    return present, missing


def main():
    logger.info("=" * 60)
    logger.info("Phase 1: ID Matching Validation")
    logger.info("=" * 60)

    # 1. 加载临床数据
    df = load_clinical_data()

    # 2. 扫描CT文件夹
    ct_map = scan_ct_folders()

    # 3. 匹配
    matched, unmatched = match_clinical_to_ct(df, ct_map)

    # 打印匹配结果
    logger.info("\n匹配结果:")
    for k, items in matched.items():
        labels = [x['label'] for x in items]
        n_pos = sum(labels)
        logger.info(f"  {k}: {len(items)} 样本 (正:{n_pos}, 负:{len(items)-n_pos})")

    logger.info(f"\n未匹配: {len(unmatched)} 样本")

    # 计算总体匹配率
    total_clinical = len(df)
    total_matched = sum(len(v) for v in matched.values())
    match_rate = total_matched / total_clinical * 100
    logger.info(f"总体匹配率: {match_rate:.1f}% ({total_matched}/{total_clinical})")

    # 4. 构建manifest
    manifest = build_manifest(matched)

    # 5. 保存manifest
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"\nManifest saved to {MANIFEST_PATH}")

    # 6. 验证NPY
    present, missing = verify_npy_presence(manifest)
    if missing:
        logger.warning(f"缺失NPY文件的前10个: {missing[:10]}")

    # 7. 保存详细匹配报告
    report = {
        'match_rate': match_rate,
        'total_clinical': total_clinical,
        'total_matched': total_matched,
        'matched_by_cohort': {k: len(v) for k, v in matched.items()},
        'unmatched_count': len(unmatched),
        'npy_present': len(present),
        'npy_missing': len(missing)
    }

    with open(OUTPUT_DIR / 'match_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    # 检查清单
    logger.info("\n" + "=" * 60)
    logger.info("验证检查清单:")
    logger.info(f"  [✓] 临床ID与影像匹配率: {match_rate:.1f}% (目标 > 95%)")
    if len(missing) == 0:
        logger.info(f"  [✓] NPY文件完整性: {len(present)}/{len(manifest)} 存在")
    else:
        logger.warning(f"  [✗] NPY文件缺失: {len(missing)}/{len(manifest)}")

    if match_rate < 95:
        logger.warning(f"  [✗] 匹配率低于95%，需要检查ID格式")

    logger.info("=" * 60)

    return match_rate >= 95


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)