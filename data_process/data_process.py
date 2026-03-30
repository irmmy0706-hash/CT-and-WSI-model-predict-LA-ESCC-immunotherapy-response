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

def data_augmentation(images):
    # 确定要旋转的切片数量，可以根据需求调整
    num_slices_to_rotate = random.randint(1, len(images))

    # 随机选择要旋转的切片的索引
    slices_to_rotate = random.sample(range(len(images)), num_slices_to_rotate)

    # 对选择的切片进行90度旋转
    for idx in slices_to_rotate:
        images[idx] = rotate_image(images[idx], 90)

    return images


def rotate_image(image, angle):
    # 将图像数组旋转指定角度（angle）（可以是90、180或270）
    rotated_image = np.rot90(image, k=angle // 90, axes=(0, 1))

    # 如果图像是RGB格式，也要调整通道的顺序
    if len(rotated_image.shape) == 3 and rotated_image.shape[2] == 3:
        rotated_image = rotated_image.transpose((1, 0, 2))

    return rotated_image




def load_images_from_folder(folder_path, img_height, img_width, channels):
    images = []

    if folder_path.endswith('.npy'):
        data = np.load(folder_path)

        if channels == 1:
            if data.shape[0] == 1:
                data = np.squeeze(data, axis=0)
            if data.ndim != 3:
                raise ValueError(f"channels=1 时，npy 应为 (n, h, w)，实际为 {data.shape}")
            for i in range(data.shape[0]):
                img = data[i]  # shape (h, w)
                img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
                img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                images.append(img)  # shape (h, w)

        elif channels == 3:
            if data.ndim != 4 or data.shape[1] != 3:
                raise ValueError(f"channels=3 时，npy 应为 (n, 3, h, w)，实际为 {data.shape}")
            for i in range(data.shape[0]):
                img = data[i]  # shape (3, h, w)
                img = np.transpose(img, (1, 2, 0))  # -> (h, w, 3)
                img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img
                img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                images.append( np.transpose(img, (2, 0, 1)))  # shape (3,h, w)

        else:
            raise ValueError("channels 只能为 1 或 3")

    else:
        raise NotImplementedError("当前版本仅支持 .npy 文件")

    return images



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


def prepare_dataset_images(dataset_path, num_classes, img_height, img_width, channels):
    class_folders = os.listdir(dataset_path)
    max_slices = 10  # 最大样切片数
    with ThreadPoolExecutor() as executor:
        sample_counts = list(
            executor.map(lambda folder: len(os.listdir(os.path.join(dataset_path, folder))), class_folders))
    set_samples = max(sample_counts)

    dataset_images = np.zeros(
        (num_classes, set_samples, max_slices, img_height, img_width, channels), dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据增强转换
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    def process_sample(class_folder, sample_folder):
        sample_path = os.path.join(dataset_path, class_folder, sample_folder)
        images = load_images_from_folder(sample_path, img_height, img_width, channels)
        # 堆叠图像成3D数组并归一化
        images = np.stack(images, axis=0).astype(np.float32) / 255.0
        # n,h,w  or  n,3,h,w
        # print(images.shape)
        tensor_data = torch.tensor(images, dtype=torch.float32).to(device)
        # 数据增强
        if channels==1:
            tensor_data = transform(tensor_data).unsqueeze(0).unsqueeze(1)
            # 1,c,n,h,w
            # print(tensor_data.shape)
        else:
            tensor_data = transform(tensor_data).unsqueeze(0).permute(0, 2, 1, 3, 4)
            # print(tensor_data.shape)

        # 统一切片数
        normalized_data = F.interpolate(tensor_data, size=(max_slices, img_height, img_width), mode='trilinear',
                                        align_corners=False)
        normalized_data = normalized_data.permute(0, 2, 3, 4, 1)
        # 1,h,w,c
        # print(normalized_data.shape)
        return normalized_data[0].cpu().numpy()

    # 使用线程池并行加载和处理每个类别的样本
    with ThreadPoolExecutor(max_workers=8) as executor:
        for i, class_folder in enumerate(class_folders):
            class_path = os.path.join(dataset_path, class_folder)
            sample_folders = os.listdir(class_path)

            futures = []
            for j, sample_folder in enumerate(sample_folders):
                futures.append(executor.submit(process_sample, class_folder, sample_folder))

            # 获取所有处理完成的结果
            results = [future.result() for future in tqdm(futures, desc=f"Processing class {class_folder}", ncols=100)]

            # 将处理后的结果存入 dataset_images
            for j, result in enumerate(results):
                dataset_images[i, j, :max_slices] = result

    return dataset_images


def process_sample(sample_path, max_slices, img_height, img_width, channels):
    images = load_images_from_folder(sample_path, img_height, img_width, channels)
    images = np.stack(images, axis=-1)  # 将切片堆叠成3D数组
    images = images.astype(np.float32) / 255.0

    tensor_data = torch.tensor(images, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(1)
    normalized_data = F.interpolate(tensor_data, size=(max_slices, img_height, img_width), mode='trilinear',
                                    align_corners=False)
    normalized_data = normalized_data.permute(0, 2, 3, 4, 1)[0].numpy()

    return normalized_data


def prepare_test_dataset_images_mul(dataset_path, num_classes, img_height, img_width, channels):
    class_folders = os.listdir(dataset_path)
    max_samples = 0

    # 并行计算最大样本数
    with ThreadPoolExecutor() as executor:
        sample_counts = list(
            executor.map(lambda folder: len(os.listdir(os.path.join(dataset_path, folder))), class_folders))
    max_samples = max(sample_counts)

    max_slices = 5  # 固定为5，优化了变量依赖

    dataset_images = np.zeros(
        (num_classes, max_samples, max_slices, img_height, img_width, channels), dtype=np.float32
    )

    # 定义多线程处理样本的任务
    for i, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_folder)
        sample_folders = os.listdir(class_path)
        num_samples = len(sample_folders)

        process_partial = partial(process_sample, max_slices=max_slices, img_height=img_height, img_width=img_width,
                                  channels=channels)

        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda sf: process_partial(os.path.join(class_path, sf)), sample_folders),
                                desc=f"Processing {class_folder}", total=num_samples, ncols=100))

        for j, processed_images in enumerate(results):
            dataset_images[i, j, :max_slices] = processed_images

    return dataset_images


def prepare_test_dataset_images_full(dataset_path, num_classes, img_height, img_width, channels):
    """ 以单线程方式处理数据集 """
    class_folders = os.listdir(dataset_path)

    # 计算最大样本数
    sample_counts = [len(os.listdir(os.path.join(dataset_path, folder))) for folder in class_folders]
    max_samples = max(sample_counts)

    max_slices = 5  # 固定为5

    # 初始化数据集数组
    dataset_images = np.zeros((num_classes, max_samples, max_slices, img_height, img_width, channels), dtype=np.float32)

    # 遍历类别文件夹
    for i, class_folder in enumerate(class_folders):
        class_path = os.path.join(dataset_path, class_folder)
        sample_folders = os.listdir(class_path)
        num_samples = len(sample_folders)

        # 如果样本不足 max_samples，重复已有样本补齐
        while len(sample_folders) < max_samples:
            sample_folders.append(sample_folders[np.random.randint(0, len(sample_folders))])

        # 逐个处理样本
        for j, sample_folder in enumerate(
                tqdm(sample_folders, desc=f"Processing {class_folder}", total=max_samples, ncols=100)):
            sample_path = os.path.join(class_path, sample_folder)
            processed_images = process_sample(sample_path, max_slices, img_height, img_width, channels)
            dataset_images[i, j, :max_slices] = processed_images

    return dataset_images


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


def creat_test_task(num_way, num_shot, num_query, dataset_images_T1,channel=1):
    import numpy as np
    from tqdm import tqdm

    num_classes, num_samples, num_slices, img_height, img_width, channels = dataset_images_T1.shape

    # 修正 batch 的计算，确保余数处理
    full_batches = num_samples // num_query
    remainder = num_samples % num_query
    total_batches = full_batches + (1 if remainder > 0 else 0)

    support_set_images_T1 = np.zeros([total_batches, num_way * num_shot, num_slices, img_height, img_width, channels],
                                     dtype=np.float32)
    query_set_images_T1 = np.zeros([total_batches, num_way * num_query, num_slices, img_height, img_width, channels],
                                   dtype=np.float32)

    support_labels = np.zeros([total_batches, num_way * num_shot], dtype=np.int32)
    query_labels = np.zeros([total_batches, num_way * num_query], dtype=np.int32)

    for i in tqdm(range(total_batches), desc="Processing few_shot batches"):
        episodic_classes = np.random.permutation(num_classes)[:num_way]

        for index, class_ in enumerate(episodic_classes):
            # 获取当前类所有样本的索引
            sample_indices = np.arange(num_samples)

            # 确定 query 样本（按顺序）
            if i == total_batches - 1 and remainder > 0:  # 最后一批且存在余数
                query_indices = sample_indices[:remainder]
                if len(query_indices) < num_query:
                    extra_samples = np.random.choice(query_indices, num_query - len(query_indices), replace=True)
                    query_indices = np.concatenate([query_indices, extra_samples])
            else:
                query_indices = sample_indices[i * num_query:(i + 1) * num_query]

            # 确定 support 样本（随机取且不与 query 重复）
            remaining_indices = np.setdiff1d(sample_indices, query_indices)
            if len(remaining_indices) < num_shot:
                extra_samples = np.random.choice(remaining_indices, num_shot - len(remaining_indices), replace=True)
                support_indices = np.concatenate([remaining_indices, extra_samples])
            else:
                support_indices = np.random.choice(remaining_indices, num_shot, replace=False)
            overlap = np.intersect1d(query_indices, support_indices)
            if len(overlap) > 0:
                print(f"Overlap found: {overlap}")

            # 填充 query 和 support
            query_set_images_T1[i, index * num_query:(index + 1) * num_query] = dataset_images_T1[class_, query_indices]
            support_set_images_T1[i, index * num_shot:(index + 1) * num_shot] = dataset_images_T1[
                class_, support_indices]

            # 更新 labels
            query_labels[i, index * num_query:(index + 1) * num_query] = [episodic_classes[index]] * num_query
            support_labels[i, index * num_shot:(index + 1) * num_shot] = [episodic_classes[index]] * num_shot

    # 转换维度，格式符合模型输入要求
    support_set_images_T1 = np.transpose(support_set_images_T1, (0, 1, 5, 2, 3, 4))
    query_set_images_T1 = np.transpose(query_set_images_T1, (0, 1, 5, 2, 3, 4))

    # 复制单通道数据到每个通道
    if channel==1:
        support_set_images_T1 = np.repeat(support_set_images_T1, 3, axis=2)
        query_set_images_T1 = np.repeat(query_set_images_T1, 3, axis=2)

    return support_set_images_T1, query_set_images_T1, support_labels, query_labels


def creat_test_task_full(num_way, num_shot, num_query, dataset_images_T1, dataset_images_support):
    import numpy as np
    from tqdm import tqdm

    num_classes, num_samples, num_slices, img_height, img_width, channels = dataset_images_T1.shape

    # 修正 batch 的计算，确保余数处理
    full_batches = num_samples // num_query
    remainder = num_samples % num_query
    total_batches = full_batches + (1 if remainder > 0 else 0)

    support_set_images_T1 = np.zeros([total_batches, num_way * num_shot, num_slices, img_height, img_width, channels],
                                     dtype=np.float32)
    query_set_images_T1 = np.zeros([total_batches, num_way * num_query, num_slices, img_height, img_width, channels],
                                   dtype=np.float32)

    support_labels = np.zeros([total_batches, num_way * num_shot], dtype=np.int32)
    query_labels = np.zeros([total_batches, num_way * num_query], dtype=np.int32)

    for i in tqdm(range(total_batches), desc="Processing few_shot batches"):
        episodic_classes = np.random.permutation(num_classes)[:num_way]

        for index, class_ in enumerate(episodic_classes):
            # 获取当前类所有样本的索引
            sample_indices = np.arange(num_samples)

            # 确定 query 样本（按顺序）
            if i == total_batches - 1 and remainder > 0:  # 最后一批且存在余数
                query_indices = sample_indices[:remainder]
                if len(query_indices) < num_query:
                    extra_samples = np.random.choice(query_indices, num_query - len(query_indices), replace=True)
                    query_indices = np.concatenate([query_indices, extra_samples])
            else:
                query_indices = sample_indices[i * num_query:(i + 1) * num_query]

            # 确定 support 样本（从 dataset_images_support 中随机取）
            support_indices = np.random.choice(sample_indices, num_shot, replace=False)

            # 填充 query 和 support
            query_set_images_T1[i, index * num_query:(index + 1) * num_query] = dataset_images_T1[class_, query_indices]
            support_set_images_T1[i, index * num_shot:(index + 1) * num_shot] = dataset_images_support[
                class_, support_indices]

            # 更新 labels
            query_labels[i, index * num_query:(index + 1) * num_query] = [episodic_classes[index]] * num_query
            support_labels[i, index * num_shot:(index + 1) * num_shot] = [episodic_classes[index]] * num_shot

    # 转换维度，格式符合模型输入要求
    support_set_images_T1 = np.transpose(support_set_images_T1, (0, 1, 5, 2, 3, 4))
    query_set_images_T1 = np.transpose(query_set_images_T1, (0, 1, 5, 2, 3, 4))

    # 复制单通道数据到每个通道
    num_channels = 3  # 需要的通道数
    support_set_images_T1 = np.repeat(support_set_images_T1, num_channels, axis=2)
    query_set_images_T1 = np.repeat(query_set_images_T1, num_channels, axis=2)

    return support_set_images_T1, query_set_images_T1, support_labels, query_labels


def few_shot(num_way, num_shot, num_query, dataset_images_T1, batch,channels=1):
    num_classes, num_samples, num_slices, img_height, img_width, channels = dataset_images_T1.shape

    support_set_images_T1 = np.zeros([batch, num_way * num_shot, num_slices, img_height, img_width, channels],
                                     dtype=np.float32)
    query_set_images_T1 = np.zeros([batch, num_way * num_query, num_slices, img_height, img_width, channels],
                                   dtype=np.float32)

    support_labels = np.zeros([batch, num_way * num_shot], dtype=np.int32)
    query_labels = np.zeros([batch, num_way * num_query], dtype=np.int32)

    for i in tqdm(range(batch), desc="Processing few_shot batches"):
        episodic_classes = np.random.permutation(num_classes)[:num_way]
        # print(episodic_classes.shape) (3,)
        for index, class_ in enumerate(episodic_classes):
            non_empty_samples = [sample for sample in dataset_images_T1[class_] if sample.any()]  # 过滤掉空的样本
            num_actual_samples = len(non_empty_samples)
            # selected 返回的是一个包含了选定样本的索引的数组。这个数组中的元素是从 0 到 num_actual_samples-1 的整数，表示了选定样本在原始数据集中的索引位置。
            selected = np.random.permutation(num_actual_samples)[:num_shot + num_query]
            selected_0 = selected[:num_shot]
            selected_1 = selected[num_shot:]

            support_set_images_T1[i, index * num_shot:(index + 1) * num_shot] = dataset_images_T1[class_, selected_0]
            query_set_images_T1[i, index * num_query:(index + 1) * num_query] = dataset_images_T1[class_, selected_1]

        s_labels = []
        q_labels = []
        for j in range(num_way):
            s_labels = s_labels + [episodic_classes[j]] * num_shot
            q_labels = q_labels + [episodic_classes[j]] * num_query
        support_labels[i] = np.array(s_labels)
        query_labels[i] = np.array(q_labels)

    support_set_images_T1 = np.transpose(support_set_images_T1, (0, 1, 5, 2, 3, 4))
    query_set_images_T1 = np.transpose(query_set_images_T1, (0, 1, 5, 2, 3, 4))

    if channels == 1:  # 想要的通道数量

        # 复制单通道数据到每个通道
        support_set_images_T1 = np.repeat(support_set_images_T1, 3, axis=2)  # (8, 3x8, 3, 5, 128, 128)
        query_set_images_T1 = np.repeat(query_set_images_T1, 3, axis=2)  # (8, 3x20, 3, 5, 128, 128)

    return support_set_images_T1, query_set_images_T1, support_labels, query_labels

if __name__ == "__main__":
    dataset_path = r'G:\BaiduNetdiskDownload\eso_data\pathology_cla\train'
    # dataset_path = r'G:\BaiduNetdiskDownload\eso_data\multimodal data\CT\train'
    num_classes = 2
    img_height = 128
    img_width = 128
    channels = 3  # 若要彩色改成 3

    processed_data = prepare_dataset_images(dataset_path, num_classes, img_height, img_width, channels)
    print(processed_data.shape)
    # 输出: (2, 186, 80, 128, 128, 1)
