#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量预处理: 将所有中心的 NIfTI/DICOM CT 数据统一转为 .npy 文件
"""
import os
import json
import glob
import numpy as np
import SimpleITK as sitk
import pandas as pd

# ============ 配置 ============
BASE_DIR = '/home/ltx/桌面/npj precision oncology'
CT_DATA_DIR = os.path.join(BASE_DIR, 'ct and wsi data')
EXCEL_PATH = os.path.join(BASE_DIR, 'clinical data', '合并队列.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessed_npy')

PADDING = 10
WINDOW_WIDTH = 400
WINDOW_CENTER = 40

# 中心名 -> 文件夹名映射
CENTER_DIR_MAP = {
    '西南': 'SW',
    '大坪': 'DP',
    '巴南': 'Banan ',  # 注意尾随空格
}


def window_transform(ct_array, windowWidth=400, windowCenter=40):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def crop_image_based_on_mask(ct_array, mask_array, padding=10):
    z_ind, y_ind, x_ind = np.where(mask_array > 0)
    if len(z_ind) == 0:
        return None

    z_min, z_max = z_ind.min(), z_ind.max()
    y_min, y_max = y_ind.min(), y_ind.max()
    x_min, x_max = x_ind.min(), x_ind.max()

    z_min = max(0, z_min - padding)
    z_max = min(ct_array.shape[0], z_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(ct_array.shape[1], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(ct_array.shape[2], x_max + padding)

    return ct_array[z_min:z_max, y_min:y_max, x_min:x_max]


def need_resample(ct_image, mask_image, tol=1e-3):
    """只在 size 不同或 spacing 差异超过容差时才需要重采样"""
    if ct_image.GetSize() != mask_image.GetSize():
        return True
    sp1 = ct_image.GetSpacing()
    sp2 = mask_image.GetSpacing()
    return any(abs(a - b) > tol for a, b in zip(sp1, sp2))


def resample_mask_to_ct(ct_image, mask_image):
    return sitk.Resample(mask_image, ct_image, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0.0, mask_image.GetPixelID())


def process_dp(patient_dir):
    """DP中心: image.nii.gz + mask.nii.gz"""
    ct_path = os.path.join(patient_dir, 'image.nii.gz')
    mask_path = os.path.join(patient_dir, 'mask.nii.gz')
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        return None

    ct_image = sitk.ReadImage(ct_path)
    mask_image = sitk.ReadImage(mask_path)

    if need_resample(ct_image, mask_image):
        mask_image = resample_mask_to_ct(ct_image, mask_image)

    ct_array = sitk.GetArrayFromImage(ct_image)
    mask_array = sitk.GetArrayFromImage(mask_image)
    return crop_image_based_on_mask(ct_array, mask_array, PADDING)


def process_dicom(patient_dir):
    """SW/Banan中心: DICOM序列 + .labels.nii mask"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(patient_dir)
    if not dicom_names:
        return None
    reader.SetFileNames(dicom_names)
    ct_image = reader.Execute()

    # 查找 .labels.nii mask
    nii_files = glob.glob(os.path.join(patient_dir, '*.labels.nii'))
    if not nii_files:
        nii_files = glob.glob(os.path.join(patient_dir, '*.nii')) + \
                    glob.glob(os.path.join(patient_dir, '*.nii.gz'))
    if not nii_files:
        return None

    mask_image = sitk.ReadImage(nii_files[0])

    if need_resample(ct_image, mask_image):
        mask_image = resample_mask_to_ct(ct_image, mask_image)

    ct_array = sitk.GetArrayFromImage(ct_image)
    mask_array = sitk.GetArrayFromImage(mask_image)
    return crop_image_based_on_mask(ct_array, mask_array, PADDING)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 读取临床数据
    df = pd.read_excel(EXCEL_PATH)
    # 排除前瞻性队列
    df = df[df['cohort'] != 'prospective'].copy()

    manifest = {}
    success, fail = 0, 0

    for _, row in df.iterrows():
        center = row['center']
        cohort = row['cohort']
        label = int(row['反应'])
        raw_id = str(row['ID'])

        # DP 的 ID 需去除 .nii.gz 后缀
        patient_id = raw_id.replace('.nii.gz', '')

        center_folder = CENTER_DIR_MAP.get(center)
        if center_folder is None:
            continue

        patient_dir = os.path.join(CT_DATA_DIR, center_folder, patient_id)
        if not os.path.isdir(patient_dir):
            print(f"[SKIP] 目录不存在: {patient_dir}")
            fail += 1
            continue

        try:
            if center == '大坪':
                cropped = process_dp(patient_dir)
            else:
                cropped = process_dicom(patient_dir)

            if cropped is None:
                print(f"[FAIL] 无法处理: {patient_id} ({center})")
                fail += 1
                continue

            # 窗宽窗位变换
            processed = window_transform(cropped, WINDOW_WIDTH, WINDOW_CENTER)

            # 保存
            save_path = os.path.join(OUTPUT_DIR, f"{patient_id}.npy")
            np.save(save_path, processed)

            manifest[patient_id] = {
                'center': center,
                'cohort': cohort,
                'label': label,
                'shape': list(processed.shape),
            }
            success += 1
            print(f"[OK] {patient_id} ({center}) shape={processed.shape}")

        except Exception as e:
            print(f"[ERROR] {patient_id} ({center}): {e}")
            fail += 1

    # 保存 manifest
    manifest_path = os.path.join(OUTPUT_DIR, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n完成! 成功: {success}, 失败: {fail}")
    print(f"Manifest 保存至: {manifest_path}")


if __name__ == '__main__':
    main()
