from model import LeNet,AlexNet,GoogleNet, ResNet18,Residual
from PIL import Image
from torchvision import transforms
import torch


image=Image.open('small5.jpg')

normalized_image=transforms.Normalize((0.18734936,0.15098935,0.09720177), (0.0698847,0.0532215,0.03424954))
test_transform=transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),normalized_image])
image=test_transform(image)
#print(image.shape)
image=image.unsqueeze(0)
#加载模型
model = GoogleNet()
model.load_state_dict(torch.load('./model/best_GoogleNet_model.pth'))
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
classes=['hotdog','no_hotdog']

with torch.no_grad():
    model.eval()
    image=image.to(device)
    output=model(image)
    pre_label=torch.max(output,1)
    result=pre_label[1].item()

print(result)
print("预测值：",classes[result])
