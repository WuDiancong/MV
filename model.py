import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from transformers import ViTForImageClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类
class BasilDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 2]  # 使用label_map列
        label = torch.tensor(label, dtype=torch.long)  # 确保标签是长整型
        label_folder = "healthy" if label == 0 else "unhealthy"

        img_path = os.path.join(self.img_dir, label_folder, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告：无法打开图片 {img_path}: {str(e)}")
            return None

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 设置路径
BASE_DIR = r"P:\image_classification_project\3.0\datasetUsed"
LABEL_PATH = os.path.join(BASE_DIR, "label.xlsx")
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "train")
VAL_IMAGE_DIR = os.path.join(BASE_DIR, "val")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test")

# 读取Excel文件
df = pd.read_excel(LABEL_PATH)
df['label_map'] = df['label_map'].astype(int)

# 分割数据集
train_df = df[df['set'] == 'train']
val_df = df[df['set'] == 'val']
test_df = df[df['set'] == 'test']

# 创建数据集
train_dataset = BasilDataset(train_df, TRAIN_IMAGE_DIR, transform=train_transform)
val_dataset = BasilDataset(val_df, VAL_IMAGE_DIR, transform=val_transform)
test_dataset = BasilDataset(test_df, TEST_IMAGE_DIR, transform=val_transform)

# 定义模型
class BasilClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(BasilClassifier, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        return self.vit(x).logits

# 初始化模型
model = BasilClassifier().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00014)

# 训练函数
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条描述
        pbar.set_description(f"Training - Loss: {loss.item():.4f}")

        # 每处理 10 个批次打印一次详细信息
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(loader)} - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条描述
            pbar.set_description(f"Validating - Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # 定义 collate_fn 来处理 None 值
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 训练循环
    num_epochs = 40
    best_val_acc = 0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        epoch_time = time.time() - start_time

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Epoch 用时: {epoch_time:.2f} 秒")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_basil_classifier.pth')
            print("保存新的最佳模型")

        print()

    # 绘制训练过程
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_basil_classifier.pth'))
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Final Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")