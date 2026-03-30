import os
import random
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from functools import partial
import cv2  # 用于图像 resize
import pickle
def data_augmentation(images):
    # 确定要旋转的切片数量，可以根据需求调整
    num_slices_to_rotate = random.randint(1, len(images))

    # 随机选择要旋转的切片的索引
    slices_to_rotate = random.sample(range(len(images)), num_slices_to_rotate)

    # 对选择的切片进行90度旋转
    for idx in slices_to_rotate:
        images[idx] = rotate_image(images[idx], 90)

    return images

def reOrganize_mDATA(mDATA):
    featGroup = []
    for tpatch in mDATA:
        tfeat = torch.from_numpy(tpatch['feature'])
        featGroup.append(tfeat.unsqueeze(0))
    featGroup = torch.cat(featGroup, dim=0)  # shape: (numPatch, 1024)
    return featGroup

def rotate_image(image, angle):
    # 将图像数组旋转指定角度（angle）（可以是90、180或270）
    rotated_image = np.rot90(image, k=angle // 90, axes=(0, 1))

    # 如果图像是RGB格式，也要调整通道的顺序
    if len(rotated_image.shape) == 3 and rotated_image.shape[2] == 3:
        rotated_image = rotated_image.transpose((1, 0, 2))

    return rotated_image



def load_images_from_folder(folder_path, pathology_path, img_height, img_width, channels):
    images = []

    # === 1. 加载 CT 图像 (.npy) ===
    if folder_path.endswith('.npy'):
        data = np.load(folder_path)

        if channels == 1:
            if data.shape[0] == 1:
                data = np.squeeze(data, axis=0)
            if data.ndim != 3:
                raise ValueError(f"channels=1 时，npy 应为 (n, h, w)，实际为 {data.shape}")
            for i in range(data.shape[0]):
                img = data[i]
                img = (img * 255).astype(np.uint8) if data.dtype != np.uint8 else img
                img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                images.append(img)

        elif channels == 3:
            if data.ndim != 4 or data.shape[1] != 3:
                raise ValueError(f"channels=3 时，npy 应为 (n, 3, h, w)，实际为 {data.shape}")
            for i in range(data.shape[0]):
                img = data[i]  # shape (3, h, w)
                img = np.transpose(img, (1, 2, 0))  # -> (h, w, 3)
                img = (img * 255).astype(np.uint8) if data.dtype != np.uint8 else img
                img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                images.append(np.transpose(img, (2, 0, 1)))  # (3, h, w)
        else:
            raise ValueError("channels 只能为 1 或 3")
    else:
        raise NotImplementedError("当前版本仅支持 .npy 文件")

    # === 2. 加载病理特征 ===
    # folder_path 可能是类似 xxx/xxx/xxx.npy → 取 .npy 前缀文件名
    base_name = os.path.splitext(os.path.basename(folder_path))[0]
    pathology_file = os.path.join(pathology_path, base_name + '.pkl')

    if os.path.exists(pathology_file):
        with open(pathology_file, 'rb') as f:
            mDATA = pickle.load(f)
        pathology_feat = reOrganize_mDATA(mDATA)  # 自定义函数，返回 torch.Tensor(n, 1024)
    else:
        # 模态缺失 → 返回零张量或 None
        pathology_feat = None

    return images, pathology_feat




def adapt_array_shape(data, target_order="nchw", current_order=None):
    """
    自适应调整 numpy 或 tensor 的大小和维度顺序到目标顺序。

    参数:
        data: np.ndarray 或 torch.Tensor，维度为3或4
        target_order: 目标维度顺序字符串，如 'nchw'，必须包含 n,c,h,w
        current_order: 输入数据当前维度顺序字符串
                       - 如果 data 是3维且为 (n,h,w)，此参数可为 None（默认）
                       - 如果 data 是4维，必须传入当前顺序，如 'nhwc'、'nchw' 等

    返回:
        调整后的数据，类型和输入一致
    """
    is_tensor = isinstance(data, torch.Tensor)
    shape = data.shape
    ndim = len(shape)

    if ndim == 3:
        # 三维数据，默认是 (n, h, w)，补充通道维度c=1
        if current_order is not None:
            raise ValueError("3维输入时，current_order 应为 None")
        # 增加channel维度，假设位置在第二维 (n, 1, h, w)
        np_data = data.cpu().numpy() if is_tensor else data
        np_data = np_data[:, None, :, :]
        current_order = "nchw"
    elif ndim == 4:
        if current_order is None:
            raise ValueError("4维输入必须提供 current_order 参数")
        np_data = data.cpu().numpy() if is_tensor else data
    else:
        raise ValueError(f"输入维度必须是3或4，当前是{ndim}")

    # 转置到目标顺序
    if current_order != target_order:
        # 构造映射
        dim_map = {k: i for i, k in enumerate(current_order)}
        permute_indices = [dim_map[d] for d in target_order]
        np_data = np.transpose(np_data, permute_indices)

    # 转回 tensor（保持原数据类型和设备）
    if is_tensor:
        return torch.tensor(np_data, dtype=data.dtype, device=data.device)
    else:
        return np_data




def prepare_dataset_images(dataset_path, pathology_path, num_classes, img_height, img_width, channels):
    class_folders = os.listdir(dataset_path)
    max_slices = 80
    with ThreadPoolExecutor() as executor:
        sample_counts = list(
            executor.map(lambda folder: len(os.listdir(os.path.join(dataset_path, folder))), class_folders))
    set_samples = max(sample_counts)
    dataset_images = np.zeros(
        (num_classes, set_samples, max_slices, img_height, img_width, channels), dtype=np.float32)

    pathology_features = [[None for _ in range(set_samples)] for _ in range(num_classes)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    def process_sample(class_folder, pathology_path, sample_folder):
        sample_path = os.path.join(dataset_path, class_folder, sample_folder)
        images, pathology_feat = load_images_from_folder(sample_path, pathology_path, img_height, img_width, channels)

        # 图像处理（堆叠 + 标准化）
        images = np.stack(images, axis=0).astype(np.float32) / 255.0
        tensor_data = torch.tensor(images, dtype=torch.float32).to(device)

        if channels == 1:
            tensor_data = transform(tensor_data).unsqueeze(0).unsqueeze(1)  # (1, C, N, H, W)
        else:
            tensor_data = transform(tensor_data).unsqueeze(0).permute(0, 2, 1, 3, 4)

        # 三线性插值到统一维度
        normalized_data = F.interpolate(
            tensor_data, size=(max_slices, img_height, img_width),
            mode='trilinear', align_corners=False
        )
        normalized_data = normalized_data.permute(0, 2, 3, 4, 1)  # (1, D, H, W, C)
        return normalized_data[0].cpu().numpy(), pathology_feat  # shape: (D, H, W, C), Tensor(n_i, 1024)

    for i, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_folder)
        sample_folders = os.listdir(class_path)

        print(f"Processing class {class_folder}".ljust(40), end='', flush=True)

        for j, sample_folder in enumerate(tqdm(sample_folders[:set_samples], desc=f"[{class_folder}]", ncols=100)):
            ct_result, path_feat = process_sample(class_folder, pathology_path, sample_folder)
            dataset_images[i, j, :max_slices] = ct_result
            pathology_features[i][j] = path_feat  # Tensor(n_i, 1024)

    return dataset_images, pathology_features



def process_sample(sample_path, max_slices, img_height, img_width, channels):
    images = load_images_from_folder(sample_path, img_height, img_width, channels)
    images = np.stack(images, axis=-1)  # 将切片堆叠成3D数组
    images = images.astype(np.float32) / 255.0

    tensor_data = torch.tensor(images, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
    normalized_data = F.interpolate(tensor_data, size=(max_slices, img_height, img_width), mode='trilinear',
                                    align_corners=False)
    normalized_data = normalized_data.permute(0, 2, 3, 4, 1)[0].numpy()

    return normalized_data


def creat_test_task(num_way, num_shot, num_query, dataset_images_T1, pathology_feat, channel=1):
    num_classes, num_samples, num_slices, img_height, img_width, channels = dataset_images_T1.shape

    full_batches = num_samples // num_query
    remainder = num_samples % num_query
    total_batches = full_batches + (1 if remainder > 0 else 0)

    support_set_images_T1 = np.zeros([total_batches, num_way * num_shot, num_slices, img_height, img_width, channels],
                                     dtype=np.float32)
    query_set_images_T1 = np.zeros([total_batches, num_way * num_query, num_slices, img_height, img_width, channels],
                                   dtype=np.float32)

    support_labels = np.zeros([total_batches, num_way * num_shot], dtype=np.int32)
    query_labels = np.zeros([total_batches, num_way * num_query], dtype=np.int32)

    # 初始化病理特征容器（列表嵌套）
    support_pathology = [[None for _ in range(num_way * num_shot)] for _ in range(total_batches)]
    query_pathology = [[None for _ in range(num_way * num_query)] for _ in range(total_batches)]

    for i in tqdm(range(total_batches), desc="Processing few_shot_multimodal batches"):
        episodic_classes = np.random.permutation(num_classes)[:num_way]

        for index, class_ in enumerate(episodic_classes):
            sample_indices = np.arange(num_samples)

            if i == total_batches - 1 and remainder > 0:
                query_indices = sample_indices[:remainder]
                if len(query_indices) < num_query:
                    extra_samples = np.random.choice(query_indices, num_query - len(query_indices), replace=True)
                    query_indices = np.concatenate([query_indices, extra_samples])
            else:
                query_indices = sample_indices[i * num_query:(i + 1) * num_query]

            remaining_indices = np.setdiff1d(sample_indices, query_indices)
            if len(remaining_indices) < num_shot:
                extra_samples = np.random.choice(remaining_indices, num_shot - len(remaining_indices), replace=True)
                support_indices = np.concatenate([remaining_indices, extra_samples])
            else:
                support_indices = np.random.choice(remaining_indices, num_shot, replace=False)

            # 填充 CT 图像
            query_set_images_T1[i, index * num_query:(index + 1) * num_query] = dataset_images_T1[class_, query_indices]
            support_set_images_T1[i, index * num_shot:(index + 1) * num_shot] = dataset_images_T1[class_, support_indices]

            # 填充 标签
            query_labels[i, index * num_query:(index + 1) * num_query] = [episodic_classes[index]] * num_query
            support_labels[i, index * num_shot:(index + 1) * num_shot] = [episodic_classes[index]] * num_shot

            # 填充 病理特征
            for j, idx in enumerate(support_indices):
                try:
                    support_pathology[i][index * num_shot + j] = pathology_feat[class_][idx]
                except:
                    support_pathology[i][index * num_shot + j] = None
            for j, idx in enumerate(query_indices):
                try:
                    query_pathology[i][index * num_query + j] = pathology_feat[class_][idx]
                except:
                    query_pathology[i][index * num_query + j] = None

    # 通道维度转换
    support_set_images_T1 = np.transpose(support_set_images_T1, (0, 1, 5, 2, 3, 4))
    query_set_images_T1 = np.transpose(query_set_images_T1, (0, 1, 5, 2, 3, 4))

    if channel == 1:
        support_set_images_T1 = np.repeat(support_set_images_T1, 3, axis=2)
        query_set_images_T1 = np.repeat(query_set_images_T1, 3, axis=2)

    return support_set_images_T1, query_set_images_T1, support_pathology, query_pathology, support_labels, query_labels

def few_shot(num_way, num_shot, num_query, dataset_images_T1, pathology_features, batch, channels=1):
    num_classes, num_samples, num_slices, img_height, img_width, channels = dataset_images_T1.shape

    support_set_images_T1 = np.zeros([batch, num_way * num_shot, num_slices, img_height, img_width, channels],
                                     dtype=np.float32)
    query_set_images_T1 = np.zeros([batch, num_way * num_query, num_slices, img_height, img_width, channels],
                                   dtype=np.float32)

    support_labels = np.zeros([batch, num_way * num_shot], dtype=np.int32)
    query_labels = np.zeros([batch, num_way * num_query], dtype=np.int32)

    # 初始化病理特征列表
    support_pathology_features = [[None for _ in range(num_way * num_shot)] for _ in range(batch)]
    query_pathology_features = [[None for _ in range(num_way * num_query)] for _ in range(batch)]

    for i in tqdm(range(batch), desc="Processing few_shot_multimodal batches"):
        episodic_classes = np.random.permutation(num_classes)[:num_way]

        for index, class_ in enumerate(episodic_classes):
            # 获取当前类别的样本索引
            non_empty_ct_samples = [k for k in range(num_samples) if dataset_images_T1[class_, k].any()]
            selected = np.random.permutation(non_empty_ct_samples)[:num_shot + num_query]
            selected_0 = selected[:num_shot]
            selected_1 = selected[num_shot:]

            # CT图像填充
            support_set_images_T1[i, index * num_shot:(index + 1) * num_shot] = dataset_images_T1[class_, selected_0]
            query_set_images_T1[i, index * num_query:(index + 1) * num_query] = dataset_images_T1[class_, selected_1]

            # 病理特征采样（保持索引一致）
            for k, sample_idx in enumerate(selected_0):
                support_pathology_features[i][index * num_shot + k] = pathology_features[class_][sample_idx]
            for k, sample_idx in enumerate(selected_1):
                query_pathology_features[i][index * num_query + k] = pathology_features[class_][sample_idx]

        # 构造标签
        s_labels = []
        q_labels = []
        for j in range(num_way):
            s_labels += [episodic_classes[j]] * num_shot
            q_labels += [episodic_classes[j]] * num_query
        support_labels[i] = np.array(s_labels)
        query_labels[i] = np.array(q_labels)

    # 转换通道为 3 通道
    support_set_images_T1 = np.transpose(support_set_images_T1, (0, 1, 5, 2, 3, 4))
    query_set_images_T1 = np.transpose(query_set_images_T1, (0, 1, 5, 2, 3, 4))

    if channels == 1:
        support_set_images_T1 = np.repeat(support_set_images_T1, 3, axis=2)
        query_set_images_T1 = np.repeat(query_set_images_T1, 3, axis=2)

    return (
        support_set_images_T1,
        query_set_images_T1,
        support_pathology_features,
        query_pathology_features,
        support_labels,
        query_labels
    )

def prepare_test_dataset_images(dataset_path, num_classes, img_height, img_width, channels):
    class_folders = os.listdir(dataset_path)
    max_samples = 0
    max_slices = 5  # 固定最大切片数为 5
    class_folder_order = []

    # 获取最大样本数
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        sample_folders = os.listdir(class_path)
        max_samples = max(max_samples, len(sample_folders))

    # 预分配 numpy 数组
    dataset_images = np.zeros(
        (num_classes, max_samples, max_slices, img_height, img_width, channels), dtype=np.float32)

    for i, class_folder in enumerate(tqdm(class_folders, desc="Processing classes", ncols=100)):
        class_path = os.path.join(dataset_path, class_folder)
        sample_folders = os.listdir(class_path)
        num_samples = len(sample_folders)  # 当前类别的样本数

        for j, sample_folder in enumerate(
                tqdm(sample_folders, desc=f"Processing {class_folder}", leave=False, ncols=80)):
            sample_path = os.path.join(class_path, sample_folder)
            class_folder_order.append(sample_folder)

            # 读取图像
            images = load_images_from_folder(sample_path, img_height, img_width, channels)
            # 堆叠图像成3D数组并归一化
            images = np.stack(images, axis=0).astype(np.float32) / 255.0
            # n,h,w  or  n,3,h,w
            # print(images.shape)
            tensor_data = torch.tensor(images, dtype=torch.float32)
            # 数据增强
            if channels == 1:
                tensor_data = (tensor_data).unsqueeze(0).unsqueeze(1)
                # 1,c,n,h,w
                # print(tensor_data.shape)
            else:
                tensor_data = (tensor_data).unsqueeze(0).permute(0, 2, 1, 3, 4)
                # print(tensor_data.shape)

            # 统一切片数
            normalized_data = F.interpolate(tensor_data, size=(max_slices, img_height, img_width), mode='trilinear',
                                            align_corners=False)
            normalized_data = normalized_data.permute(0, 2, 3, 4, 1)
            # 1,h,w,c
            # print(normalized_data.shape)

            # 存入 dataset_images
            dataset_images[i, j, :max_slices] = normalized_data[0].numpy()

    return dataset_images, class_folder_order
# if __name__ == "__main__":
#     pathology_path =
#     dataset_path =
#     num_classes = 2
#     img_height = 128
#     img_width = 128
#     channels = 1  # 若要彩色改成 3
#
#     processed_data,pathology_features = prepare_dataset_images(dataset_path,pathology_path, num_classes, img_height, img_width, channels)
#     (s_ct, q_ct, s_path, q_path, s_labels, q_labels) = few_shot(
#         num_way=2, num_shot=5, num_query=15,
#         dataset_images_T1=processed_data,
#         pathology_features=pathology_features,
#         batch=8
#     )
#     for index in tqdm(range(8), desc="training batches", ncols=100):
#         support_T1_data = torch.tensor(s_ct[index])
#         query_T1_data = torch.tensor(q_ct[index])
#         support_la = torch.tensor(s_labels[index])
#         query_la = torch.tensor(q_labels[index])
#         print(support_T1_data.shape,query_T1_data.shape)
#         support_path_feat = s_path[index]  # List of length (num_way * num_shot), each is Tensor(n_i, 1024)
#         query_path_feat = q_path[index]    # List of length (num_way * num_query)
#         for k, feat in enumerate(support_path_feat):
#             if feat is not None:
#                 print(f"Support sample {k} 病理特征 shape: {feat.shape}")  # (n_i, 1024)
#             else:
#                 print(f"Support sample {k} 病理特征缺失")