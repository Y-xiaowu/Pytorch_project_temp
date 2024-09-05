import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class LeNet(nn.Module):#28*28
    def __init__(self):
        super(LeNet,self).__init__()

        self.conv1=nn.Conv2d(1,6,5,stride=1,padding=2)#input channel=1,output channel=6,kernel size=5
        self.sigmoid=nn.Sigmoid()
        self.pool1=nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv2=nn.Conv2d(6,16,5,stride=1,padding=0)
        self.pool2=nn.AvgPool2d(kernel_size=2,stride=2)

        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(400,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.sigmoid(x)
        x=self.pool1(x)

        x=self.conv2(x)
        x=self.sigmoid(x)
        x=self.pool2(x)

        x=self.flatten(x)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)

        return x

class AlexNet(nn.Module):#227*227
    def __init__(self):
        super(AlexNet,self).__init__()
        self.relu = nn.ReLU()
        self.conv1=nn.Conv2d(1,96,11,stride=4,)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2=nn.Conv2d(96,256,5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3=nn.Conv2d(256,384,3,padding=1)
        self.conv4=nn.Conv2d(384,384,3,padding=1)
        self.conv5=nn.Conv2d(384,256,3,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(9216,4096)
        self.fc2=nn.Linear(4096,4096)
        self.fc3=nn.Linear(4096,10)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool3(x)

        x=self.flatten(x)
        x=self.relu(self.fc1(x))
        x=F.dropout(x,0.6)
        x=self.relu(self.fc2(x))
        x=F.dropout(x,0.6)
        x=self.fc3(x)

        return x


class VGG16(nn.Module):#224*224
    def __init__(self):
        super(VGG16,self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(1,64,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.block2=nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.block3=nn.Sequential(
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.block4=nn.Sequential(
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.block5=nn.Sequential(
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        self.block6=nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7,2048),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(2048,1000))

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.block6(x)
        return x

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4):
        super(Inception,self).__init__()
        self.relu=nn.ReLU()
        # 1*1 conv
        self.p1_1=nn.Conv2d(in_channels,out_channels=c1,kernel_size=1)
        # 3*3 conv
        self.p2_1=nn.Conv2d(in_channels,out_channels=c2[0],kernel_size=1)
        self.p2_2=nn.Conv2d(c2[0],out_channels=c2[1],kernel_size=3,padding=1)
        # 5*5 conv
        self.p3_1=nn.Conv2d(in_channels,out_channels=c3[0],kernel_size=1)
        self.p3_2=nn.Conv2d(c3[0],out_channels=c3[1],kernel_size=5,padding=2)
        # 3*3 maxpool
        self.p4_1=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2=nn.Conv2d(in_channels,out_channels=c4,kernel_size=1)


    def forward(self,x):
        p1=self.relu(self.p1_1(x))
        p2=self.relu(self.p2_2(self.relu(self.p2_1(x))))
        p3=self.relu(self.p3_2(self.relu(self.p3_1(x))))
        p4=self.relu(self.p4_2(self.p4_1(x)))

        #print(p1.shape,p2.shape,p3.shape,p4.shape)

        return torch.cat([p1,p2,p3,p4],dim=1)


class GoogleNet(nn.Module):#224*224
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.block2=nn.Sequential(
            nn.Conv2d(64,64,1), # 64*224*224
            nn.ReLU(),
            nn.Conv2d(64,192,3,padding=1), # 192*224*224
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.block3=nn.Sequential(
            Inception(192,64,(96,128),(16,32),32),
            Inception(256,128,(128,192),(32,96),64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.block4=nn.Sequential(
            Inception(480,192,(96,208),(16,48),64),
            Inception(512,160,(112,224),(24,64),64),
            Inception(512,128,(128,256),(24,64),64),
            Inception(512,112,(128,288),(32,64),64),
            Inception(528,256,(160,320),(32,128),128),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.block5=nn.Sequential(
            Inception(832,256,(160,320),(32,128),128),
            Inception(832,384,(192,384),(48,128),128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(1024,2))

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):#224*224
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        return x


class Residual(nn.Module):
    def __init__(self,in_channels,num_channels,use_1x1conv=False,stride=1):
        super(Residual,self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=num_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3=nn.Conv2d(in_channels=in_channels,out_channels=num_channels,kernel_size=1,stride=stride)
        else:
            self.conv3=None

    def forward(self,x):
        y=self.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x=self.conv3(x)
        y=self.relu(x+y)
        return y


class ResNet18(nn.Module):
    def __init__(self,Residual):
        super(ResNet18,self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(1,64,7,stride=2,padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        self.block2=nn.Sequential(
            Residual(64,64,False,1),
            Residual(64,64,False,1))
        self.block3=nn.Sequential(
            Residual(64,128,True,2),
            Residual(128,128,False,1))
        self.block4=nn.Sequential(
            Residual(128,256,True,2),
            Residual(256,256,False,1))
        self.block5=nn.Sequential(
            Residual(256,512,True,2),
            Residual(512,512,False,1))
        self.block6=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512,10))
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)



    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.block6(x)
        return x


if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #model=LeNet().to(device)
    #model=AlexNet().to(device)
    #model=VGG16().to(device)
    #model=GoogleNet().to(device)
    model=ResNet18(Residual).to(device)
    summary(model,(1,224,224))
    print(model)














