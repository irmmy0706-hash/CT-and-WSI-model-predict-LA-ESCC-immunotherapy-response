import pandas as pd
import numpy as np
# 必须显式导入 enable_iterative_imputer 才能使用 IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

# ================= 配置路径 =================
# 使用 r 前缀防止 Windows 路径中的反斜杠被转义
input_path = r"H:\edge donwnload\clinical.csv"
output_dir = r"H:\edge donwnload"  # 输出文件将保存在这里

# 1. 读取数据
try:
    df = pd.read_csv(input_path)
    print(f"成功读取文件: {input_path}")
except FileNotFoundError:
    print(f"错误：无法找到文件，请确认路径是否正确: {input_path}")
    exit()

# 2. 定义变量类型
# 分类/等级变量 (使用众数填充)
categorical_cols = ['ECOG', 'Smoking', 'Drinking', 'cT', 'cN', 'cM']
# 连续数值变量 (需要高级填充)
continuous_cols_to_impute = ['Alb', 'WBC', 'HBG']
# 辅助变量 (用于计算相关性)
feature_cols = ['age', 'BMI', 'Alb', 'WBC', 'HBG', 'NEU', 'LNM']

# ==========================================
# 准备基础数据 (填充分类变量)
# ==========================================
# 无论用哪种高级方法填充数值，分类变量我们都统一用"众数"先填好
df_base = df.copy()
for col in categorical_cols:
    if col in df_base.columns:
        mode_val = df_base[col].mode()[0]
        df_base[col] = df_base[col].fillna(mode_val)

# ==========================================
# 方法一：基础填充 (数值用中位数)
# ==========================================
df_basic = df_base.copy()
for col in continuous_cols_to_impute:
    median_val = df_basic[col].median()
    df_basic[col] = df_basic[col].fillna(median_val)

output_basic = f"{output_dir}\\clinical_basic_imputed.csv"
df_basic.to_csv(output_basic, index=False)
print(f"1. 基础填充文件已保存: {output_basic}")

# ==========================================
# 方法二：KNN (K近邻) 填充
# ==========================================
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = df_base.copy()
# 只对数值相关的列进行计算和替换
df_numeric_knn = df_knn[feature_cols].copy()
df_knn[feature_cols] = knn_imputer.fit_transform(df_numeric_knn)

output_knn = f"{output_dir}\\clinical_knn_imputed.csv"
df_knn.to_csv(output_knn, index=False)
print(f"2. KNN填充文件已保存: {output_knn}")

# ==========================================
# 方法三：MICE (多重插补) 填充 - 推荐
# ==========================================
mice_imputer = IterativeImputer(random_state=42)
df_mice = df_base.copy()
df_numeric_mice = df_mice[feature_cols].copy()
df_mice[feature_cols] = mice_imputer.fit_transform(df_numeric_mice)

output_mice = f"{output_dir}\\clinical_mice_imputed.csv"
df_mice.to_csv(output_mice, index=False)
print(f"3. MICE填充文件已保存: {output_mice}")

print("\n全部完成！请去 H:\\edge donwnload\\ 文件夹查看结果。")