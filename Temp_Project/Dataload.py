import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

def train_val_dataloader():

    root_dir = './data'
    batch_size = 32

    tf = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    #加载数据集
    train_data = ImageFolder(root=root_dir+'/train',transform=tf)
    val_data = ImageFolder(root=root_dir+'/val',transform=tf)

    #定义数据加载器
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size,shuffle=False)

    return train_loader,val_loader