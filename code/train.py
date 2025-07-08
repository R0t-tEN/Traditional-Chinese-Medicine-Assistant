import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm # 引入tqdm

# --- 1. 定义超参数和配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_DIR = './dataset' # 你的数据集路径，假设和脚本在同一目录下
NUM_EPOCHS = 25       # 训练的总轮数
BATCH_SIZE = 16       # 每批次训练的图片数量
LEARNING_RATE = 0.001 # 学习率

# --- 2. 数据预处理和加载 ---
# 定义数据增强和转换
# 训练集需要数据增强，验证集不需要
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(),             # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),            # 缩放到256
        transforms.CenterCrop(224),        # 中心裁剪到224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 使用ImageFolder加载数据

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

print(f"识别的类别共 {num_classes} 种, 它们是: {class_names}")
print(f"训练集图片数量: {dataset_sizes['train']}")
print(f"验证集图片数量: {dataset_sizes['val']}")

# --- 3. 定义模型 ---
# 加载预训练的ResNet18模型
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# 获取ResNet的全连接层输入特征数
num_ftrs = model.fc.in_features

# 替换为我们自己的全连接层，以匹配我们的类别数
model.fc = nn.Linear(num_ftrs, num_classes)

# 将模型移动到指定的设备 (GPU or CPU)
model = model.to(DEVICE)

# --- 4. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数，适用于多分类问题
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Adam优化器

# --- 5. 训练和验证循环 ---
def train_model(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm创建进度条
            progress_bar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()} Epoch {epoch+1}')

            # 遍历数据
            for inputs, labels in progress_bar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 后向传播 + 仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # 更新tqdm进度条的后缀信息
                progress_bar.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item() / inputs.size(0))


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

    return model

# --- 6. 开始训练 ---
if __name__ == '__main__':
    model_trained = train_model(model, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    # --- 7. 保存训练好的模型 ---
    # 我们只保存模型的参数（权重）
    print("训练完成，正在保存模型...")
    torch.save(model_trained.state_dict(), 'tcm_model_resnet18.pth')
    print("模型已保存为 tcm_model_resnet18.pth")