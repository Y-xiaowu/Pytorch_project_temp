import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from model import LeNet,AlexNet,ResNet18,Residual

batch_size = 1

def train_val_data_process():
    # Load the dataset
    #tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((28, 28))])
    tf = transforms.Compose([transforms.ToTensor(), transforms.Resize((227, 227))])
    test_data = FashionMNIST(root='./data', train=False, transform=tf, download=True)

    # Create data loaders
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)

    return test_loader

def test_model_process(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_num = 0
    test_correct = 0
    model.eval()
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            pre_label = torch.argmax(output, 1)

            result = pre_label.item()
            labelss = label.item()
            print('Test Image: {} | Label: {} | Predict: {}'.format(test_num, labelss, result))

            test_correct += torch.sum(pre_label == label.data)
            test_num += image.size(0)

    test_acc = test_correct.double().item() / test_num

    print('Test Accuracy: {:.2f}%'.format(100 * test_acc))

if __name__ == '__main__':
    test_loader = train_val_data_process()
    #model = LeNet()
    #model = AlexNet()
    model = ResNet18(Residual)
    model.load_state_dict(torch.load('./model/best_ResNet18_model.pth'))
    test_model_process(model, test_loader)
    print('Model test finished.')

