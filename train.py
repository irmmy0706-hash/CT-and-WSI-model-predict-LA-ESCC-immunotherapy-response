import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_process.data_process import prepare_dataset_images, few_shot
from data_process.data_process_mul import prepare_dataset_images as prepare_dataset_images_mul, few_shot as few_shot_mul
from metric import acc_, f1_score_, precision_, recall_
from model.DTMF import MergeModel as mulmodel  # 导入你的自定义PyTorch模型
from model.model_3d import MergeModel as singlemodel  # 导入你的自定义PyTorch模型

import pickle


warnings.filterwarnings("ignore")

# 设置随机种子以保持结果的一致性
torch.manual_seed(42)

num_batch = 50 #crop_calss2
gpuid = 2
num_epochs = 1
inner_train_step = 5
update_test_step = 3
learning_rate_inner = 0.001
learning_rate_outer = 0.001

# 图像的长、宽、通道数
num_classes = 2  # 类别数量
img_height = 128
img_width = 128
channels = 1

num_shot = 5  # 支持集样本数,crop_class2
num_query = 5  # 查询集样本数,crop_class2

root_dir='CTA_only'
pathology_path = 'G:\eso_data\multimodal data\pathology_feature'
dataset_path1 = r'G:\eso_data\multimodal data\CT\train'
# 新增：用于存储特征的字典
patient_features = {}
patient_labels = {}
is_multimodal= True
os.makedirs(root_dir, exist_ok=True)

def save_data_to_cache(cache_path, data):
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_data_from_cache(cache_path):
    with open(cache_path, 'rb') as f:
        return pickle.load(f)
# 准备数据集
cache_path = os.path.join(root_dir,'dataset.pkl')
if os.path.exists(cache_path):
    dataset_images1 = load_data_from_cache(cache_path)
else:
    if is_multimodal:
        dataset_images1, pathology_features = prepare_dataset_images_mul(dataset_path1, pathology_path, num_classes, img_height, img_width, channels)
    else:
        dataset_images1 = prepare_dataset_images(dataset_path1, num_classes, img_height, img_width, channels)

    # save_data_to_cache(cache_path, dataset_images1)


def are_models_identical(model1, model2):
    # 获取模型1和模型2的状态字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 检查两个模型的参数数量是否相同
    if len(state_dict1) != len(state_dict2):
        return False

    # 逐个参数比较它们是否相同
    for key1, value1 in state_dict1.items():
        if key1 in state_dict2:
            value2 = state_dict2[key1]
            if not torch.allclose(value1, value2):
                return False
        else:
            return False

    return True


# 判断meta_weights和after_weight是否一致
def are_model_weights_identical(weights1, weights2):
    for key in weights1.keys():
        if torch.allclose(weights1[key], weights2[key]):
            continue
        else:
            return False
    return True

def load_model_and_epoch(model, inner_optimizer,outer_optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型权重
        inner_optimizer.load_state_dict(checkpoint['inner_optimizer_state_dict'])  # 加载优化器状态（如果有）
        outer_optimizer.load_state_dict(checkpoint['outer_optimizer_state_dict'])  # 加载优化器状态（如果有）
        epoch = checkpoint['epoch']  # 加载 epoch
        loss = checkpoint['loss']  # 可选：加载保存的损失
        print(f"Resuming training from epoch {epoch}...")
    else:
        epoch = 0  # 如果没有权重文件，训练从 epoch 0 开始
        loss = None
        print("No checkpoint found, starting from scratch.")

    return model, inner_optimizer,outer_optimizer, epoch, loss
def save_model_and_epoch(model, inner_optimizer,outer_optimizer, epoch, loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'inner_optimizer_state_dict': inner_optimizer.state_dict(),
        'outer_optimizer_state_dict': outer_optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Model and epoch {epoch} saved at {checkpoint_path}")

def train_step(model, inner_optimizer, outer_optimizer, support_set_images_T1, query_set_images_T1, support_pathology_features,query_pathology_features,support_labels,
               query_labels, batch,is_mul=True):
    train_loss = []
    train_precision = []
    train_acc = []
    train_recall = []
    train_f1 = []

    # 备份初始模型参数
    meta_weights = model.state_dict()
    # meta_weights_hash = hashlib.md5(str(meta_weights).encode()).hexdigest()
    # print(f"meta_weights 的哈希值: {meta_weights_hash}")

    for index in tqdm(range(batch), desc="training batches", ncols=100):
        model.load_state_dict(meta_weights)
        support_T1_data = torch.tensor(support_set_images_T1[index]).to(device)
        query_T1_data = torch.tensor(query_set_images_T1[index]).to(device)
        support_la = torch.tensor(support_labels[index]).to(device)
        query_la = torch.tensor(query_labels[index]).to(device)
        if is_mul:
            support_path_feat = support_pathology_features[index]  # List of length (num_way * num_shot), each is Tensor(n_i, 1024)
            query_path_feat = query_pathology_features[index]  # List of length (num_way * num_query)
            query_path_feat = [t.to(device) if t is not None else None for t in query_path_feat]
            support_path_feat = [t.to(device) if t is not None else None for t in support_path_feat]
        for inner_step in range(inner_train_step):
            # print(support_T1_data.shape)
            support_logits,_ = model(support_T1_data,support_path_feat) if is_mul else model(support_T1_data)
            support_la = support_la.to(torch.long)
            support_loss = criterion(support_logits, support_la)

            # 反向传播和参数更新

            inner_optimizer.zero_grad()
            support_loss.backward()
            inner_optimizer.step()

        query_logits,_ = model(query_T1_data,query_path_feat) if is_mul else model(query_T1_data)
        query_la = query_la.to(torch.long)
        query_loss = criterion(query_logits, query_la)
        query_pred = torch.softmax(query_logits, dim=1)

        # 计算评估指标
        # print(f"Labels: {query_la}")
        # print(f"Preds: {torch.argmax(query_pred, dim=-1)}")
        epoch_acc = acc_(query_la, torch.argmax(query_pred, dim=-1))
        epoch_precision = precision_(query_la, torch.argmax(query_pred, dim=-1))
        epoch_recall = recall_(query_la, torch.argmax(query_pred, dim=-1))
        epoch_f1 = f1_score_(query_la, torch.argmax(query_pred, dim=-1))

        train_loss.append(query_loss.item())
        train_acc.append(epoch_acc)
        train_precision.append(epoch_precision)
        train_recall.append(epoch_recall)
        train_f1.append(epoch_f1)

    # 计算元训练损失
    train_loss = [torch.tensor(loss) for loss in train_loss]
    stacked_losses = torch.stack(train_loss)
    # 计算堆叠后的损失张量的总和
    meta_batch_loss = torch.sum(stacked_losses)
    meta_batch_loss.requires_grad = True

    model.load_state_dict(meta_weights)

    outer_optimizer.zero_grad()
    # 梯度下降更新模型参数
    meta_batch_loss.backward()
    outer_optimizer.step()

    # after_weight = model.state_dict()
    # after_weights_hash = hashlib.md5(str(after_weight).encode()).hexdigest()
    # print(f"训练after_weights 的哈希值: {after_weights_hash}")

    every_loss = [loss.numpy() for loss in train_loss]
    every_acc = [acc.cpu().numpy() for acc in train_acc]
    every_precision = [precision for precision in train_precision]
    every_recall = [recall for recall in train_recall]
    every_f1 = [f1 for f1 in train_f1]

    return meta_batch_loss, every_loss, every_acc, every_precision, every_recall, every_f1


best_test_acc = 0.0
best_test_pred = 0.0

# 存储每个epoch的测试指标
test_metrics = []
device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
print(f"Using GPU: {gpuid}" if torch.cuda.is_available() else "CUDA not available, using CPU")
# 创建模型
if is_multimodal:
    Model = mulmodel(num_classes)
else:
    Model = singlemodel(num_classes)

Model = Model.to(device)  # 将模型移到GPU

# 创建损失函数和优化器
criterion = nn.CrossEntropyLoss()
inner_optimizer = optim.SGD(Model.parameters(), lr=learning_rate_inner, weight_decay=1e-3, momentum=0.9)
outer_optimizer = optim.SGD(Model.parameters(), lr=learning_rate_outer, weight_decay=1e-3, momentum=0.9)


# 打开日志文件，如果文件不存在则会创建
log_file = open(os.path.join(root_dir, 'train_log.txt'), 'w')

Model, inner_optimizer,outer_optimizer, start_epoch, loss = load_model_and_epoch(Model, inner_optimizer,outer_optimizer, os.path.join(root_dir,f'{num_shot}_{num_query}_class2_latest.pth'))

for epoch in range(start_epoch,num_epochs):
    average_loss = 0.0
    average_accuracy = 0.0
    average_precision = 0.0
    average_recall = 0.0
    average_f1 = 0.0

    # 训练

    # 在每个 epoch 中生成支持集和查询集
    if is_multimodal:
        (support_set_images1_epoch, query_set_images1_epoch,support_pathology_features,query_pathology_features, support_labels1_epoch, query_labels1_epoch) = few_shot_mul(
        num_classes, num_shot, num_query, dataset_images1,pathology_features, batch=num_batch,channels=channels)
    else:
        support_set_images1_epoch, query_set_images1_epoch,support_labels1_epoch, query_labels1_epoch = few_shot(
            num_classes, num_shot, num_query, dataset_images1, batch=num_batch, channels=channels)
        support_pathology_features=None
        query_pathology_features=None

    # 在当前 epoch 中只调用一次 train_step
    support_set_images1_batch = support_set_images1_epoch
    query_set_images1_batch = query_set_images1_epoch
    support_labels1_batch = support_labels1_epoch
    query_labels1_batch = query_labels1_epoch

    meta_batch_loss, every_loss, every_acc, every_precision, every_recall, every_f1 = train_step(Model,
                                                                                                 inner_optimizer,
                                                                                                 outer_optimizer,
                                                                                                 support_set_images1_batch,
                                                                                                 query_set_images1_batch,
                                                                                                 support_pathology_features,
                                                                                                 query_pathology_features,
                                                                                                 support_labels1_batch,
                                                                                                 query_labels1_batch,
                                                                                                 num_batch,
                                                                                                 is_multimodal)

    # model_after_train = MergeModel(num_classes)
    # model_after_train.to(device)
    # model_after_train.load_state_dict(Model.state_dict())  # 复制当前模型的参数
    if (epoch + 1) % 10 == 0:
        save_model_and_epoch(Model, inner_optimizer,outer_optimizer, epoch, average_loss,
                             os.path.join(root_dir,f'{num_shot}_{num_query}_class2_epoch_{epoch + 1}.pth'))

    save_model_and_epoch(Model, inner_optimizer, outer_optimizer, epoch, average_loss,
                         os.path.join(root_dir,f'{num_shot}_{num_query}_class2_latest.pth'))

    average_loss += np.mean(every_loss)
    average_accuracy += np.mean(every_acc)
    average_precision += np.mean(every_precision)
    average_recall += np.mean(every_recall)
    average_f1 += np.mean(every_f1)

    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"Average Loss: {average_loss:.8f}")
    print(f"Average Accuracy: {average_accuracy:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Average Recall: {average_recall:.4f}")
    print(f"Average F1 Score: {average_f1:.4f}")

    log_file.write(f"Epoch {epoch + 1}/{num_epochs}:\n")
    log_file.write(f"Average Loss: {average_loss:.8f}\n")
    log_file.write(f"Average Accuracy: {average_accuracy:.4f}\n")
    log_file.write(f"Average Precision: {average_precision:.4f}\n")
    log_file.write(f"Average Recall: {average_recall:.4f}\n")
    log_file.write(f"Average F1 Score: {average_f1:.4f}\n")


log_file.close()
