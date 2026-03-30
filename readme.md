# Multi-Modal Learning Framework for Pathology and CTA Data

This repository provides the implementation for a multi-modal deep learning framework using pathology and CTA (Computed Tomography Angiography) data. It supports both single-modal and multi-modal training and evaluation tasks.

> **Note**  
> The patch extraction script (`gen_patch_noLabel_stride_MultiProcessing_multiScales.py`) is adapted from the DTFD-MIL project:  
> *"DTFD-MIL: Double-Tier Feature Distillation Multiple Instance Learning for Histopathology"*  
> GitHub: [https://github.com//DTFD-MIL](https://github.com//DTFD-MIL)  
> This script is used solely for data preprocessing and is not part of the original contribution of this work.

---

## 🚀 Training Steps

1. **Patch Extraction (Pathology)**  
   Use the following script to extract patches from `.svs` whole-slide images:
   `python gen_patch_noLabel_stride_MultiProcessing_multiScales.py`
   
2. **Feature Extraction (Pathology)**
   Extract deep features from the patchified images
   `python main_Extract_PerSlide.py`
3. **Model Training**
Run the training script:
`python train.py`
Set the following parameters:
	`pathology_path`: Path to the extracted pathology features.
	`dataset_path1`: Path to the CTA feature dataset.
	Training mode options:
	`Multi-modal`: is_multimodal = True
	`Single-modal`: is_multimodal = False

##🧪 Testing Steps

1. **CTA single-modal testing:**
	`python test_CTA.py`

1. **Pathology single-modal testing:**
	`python test_pathology.py`

1. **Multi-modal testing:**
	`python test_multimodal.py`

##📁 Dataset Format

1. **Single-Modal Input Format**
dataset/
├── train/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/
	
2. **Each subdirectory contains .npy feature files corresponding to class labels (e.g., 0 or 1).**
pathology/
├── case_001.npy
├── case_002.npy
├── ...
Each .npy file contains the feature representation of one whole-slide image.
CTA/
├── train/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/





