from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set the parameters
batch_size = 64

# Load the dataset
tf=transforms.Compose([transforms.ToTensor(), transforms.Resize((28,28))])
train_data = FashionMNIST(root='./data', train=True, transform=tf,download=True)
test_data = FashionMNIST(root='./data', train=False, transform=tf,download=True)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#get one batch of data
for step,(x,y) in enumerate(train_loader):
    if step>0:
        break
    batch_x = x.squeeze().numpy()#图片#将四维张量移除第一维，并转化为numpy数组
    batch_y = y.numpy()#标签
    class_labels=train_data.classes#类别标签
    # print(class_labels)
    # print("batch_x shape:", batch_x.shape)
    # print("batch_y shape:", batch_y.shape)

# Visualize the data of one batch
plt.figure(figsize=(12, 5))#设置画布大小
for img in np.arange(len(batch_x)):#循环遍历每一张图片
    plt.subplot(4, 16, img + 1)##创建子图
    plt.imshow(batch_x[img], cmap='gray')#显示图片
    plt.title(class_labels[batch_y[img]],size=6)#显示标签
    plt.axis('off')#关闭坐标轴
    plt.subplots_adjust(wspace=0.1)#调整子图间距
plt.show()