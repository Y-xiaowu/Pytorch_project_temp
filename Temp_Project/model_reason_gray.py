from model import LeNet,AlexNet, ResNet18,Residual
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2

image=Image.open('cat1.jpg')

gray_image = image.convert('L')

# 将PIL图像转换为NumPy数组
gray_array = np.array(gray_image)

# 归一化灰度图像到[0, 1]范围
normalized_gray_array = gray_array / 255.0

# 如果需要，可以将归一化的NumPy数组转换回PIL图像
normalized_gray_image = Image.fromarray((normalized_gray_array * 255).astype(np.uint8))

normalized_image=transforms.Normalize(mean=[0.45], std=[0.225])
test_transform=transforms.Compose([transforms.Resize((224,224)),transforms.CenterCrop(224),
                                   transforms.ToTensor(),normalized_image])
image=test_transform(normalized_gray_image)
#print(image.shape)

image=image.unsqueeze(0)#添加一个维度


#加载模型
model = ResNet18(Residual)
model.load_state_dict(torch.load('./model/best_ResNet18_model1.pth'))
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
classes=['cat','dog','horse','sheep','zebra' ,'giraffe','elephant','bear','lion','tiger']

with torch.no_grad():
    model.eval()
    image=image.to(device)
    output=model(image)
    pre_label=torch.max(output,1)
    result=pre_label[1].item()

print(result)
print("预测值：",classes[result])
