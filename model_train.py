import time
import copy

import pandas as pd
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from model import LeNet, AlexNet, VGG16,GoogleNet, ResNet18,Residual
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import ImageFolder

# defining Hyperparameters
batch_size = 32
num_epochs = 20
learning_rate = 0.001

#root_dir = './data/flower_photos'

def train_val_data_process():
    # Load the dataset
    #tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
    #tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((227, 227))])
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    train_data = FashionMNIST(root='./data', train=True, transform=tf, download=True)
    train_data, val_data = torch.utils.data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader, val_loader

# def train_val_data_process():
#     # Load the dataset
#     tf = transforms.Compose(
#         [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
#     train_data = ImageFolder(root=root_dir+'/train',transform=tf)
#     train_data, val_data = torch.utils.data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
#
#     # Create data loaders
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2)
#
#     return train_loader, val_loader

def train_model_process (model, train_loader, val_loader, num_epochs=10):
    print("Training the model...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    best_model_wts = copy.deepcopy(model.state_dict())

    #初始化参数
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        val_loss = 0.0
        train_correct = 0
        val_correct = 0

        train_num = 0
        val_num = 0

        for step, (input, label) in enumerate(train_loader):
            input = input.to(device)
            label= label.to(device)

            model.train()
            outputs = model(input)
            pre_label = torch.argmax(outputs, 1)
            loss = criterion(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input.size(0)
            train_correct += torch.sum(pre_label == label.data)
            train_num += input.size(0)


        scheduler.step()     # 更新学习率

        for step, (input, label) in enumerate(val_loader):
            input = input.to(device)
            label= label.to(device)

            model.eval()
            outputs = model(input)
            pre_label = torch.argmax(outputs, 1)
            loss = criterion(outputs, label)

            val_loss += loss.item() * input.size(0)
            val_correct +=torch.sum (pre_label == label.data)
            val_num += input.size(0)

        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_correct.double().item()/train_num)
        val_acc_all.append(val_correct.double().item()/val_num)
        print('{} Train Loss: {:.4f}, Acc: {:.4f}, Val Loss: {:.4f}, Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1], val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), './model/best_ResNet18_model.pth')
            print("save best model")

        time_used = time.time() - since
        print('Best val Acc: {:4f},learning rate: {:6f},Training complete in {:.0f}m {:.0f}s'.format(best_acc, scheduler.get_last_lr()[0],time_used // 60, time_used % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './model/best_ResNet18_model_final.pth')

    train_process=pd.DataFrame({"epoch":range(num_epochs),
                                     "train_loss":train_loss_all,
                                     "train_acc":train_acc_all,
                                     "val_loss":val_loss_all,
                                     "val_acc":val_acc_all })

    return train_process


def matplotlib_loss_acc_plot(train_process):

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process['train_loss'],'ro-', label='train_loss')
    plt.plot(train_process['epoch'], train_process['val_loss'],'bs-', label='val_loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process['train_acc'],'ro-', label='train_acc')
    plt.plot(train_process['epoch'], train_process['val_acc'],'bs-', label='val_acc')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.show()

if __name__ == '__main__':
    # Load the dataset
    train_loader, val_loader = train_val_data_process()
    #model = LeNet()
    #model= AlexNet()
    #model = VGG16()
    #model = GoogleNet()
    model = ResNet18(Residual)

#   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#   model.load_state_dict(torch.load('./model/best_VGG16_model.pth', map_location=device))

    print(model)
    train_process = train_model_process(model, train_loader, val_loader, num_epochs=num_epochs)
    matplotlib_loss_acc_plot(train_process)