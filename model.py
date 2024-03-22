import matplotlib.pyplot as plt
import random
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm import tqdm


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_64 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linA = nn.Linear(in_features=4608, out_features=10)
        
        self.softmax = nn.Softmax(dim=0)
        
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_64(x)
        x = self.ReLU(x)
        x = self.maxpool(x)
        x = self.ReLU(x)
        
        x = self.conv_128(x)
        x = self.ReLU(x)
        x = self.maxpool(x)
        x = self.ReLU(x)

        print(x.shape)
        
        x = x.flatten()
        x = self.linA(x)

        return x


to_tensor = ToTensor()
train = CIFAR10('.', train=True, transform=to_tensor)

if __name__ == '__main__':
    loader = DataLoader(train, drop_last=True)

    model = Classifier()

    opt = SGD(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    losses = []
    for e in tqdm(range(50)):
        iter_loss = 0
        for i, data in enumerate(tqdm(loader, leave=False)):
            image, label = data

            target = torch.empty(10)
            target[label] = 1.

            opt.zero_grad()

            pred = model(image)

            loss = loss_function(pred, target)
            loss.backward()
            iter_loss += loss.item()

            opt.step()
            
        losses.append(iter_loss)
        
    torch.save(model, 'classifier.pt')
    plt.plot(losses)
    plt.show()


test = CIFAR10('.', train=False, transform=to_tensor)
# predictions = torch.Tensor([torch.argmax(model(test[i][0].unsqueeze(0))) for i in range(len(test))])
# truths = torch.Tensor([test[i][1] for i in range(len(test))])
# print(f'test accuracy: {torch.sum(predictions == truths) / len(truths)}')

def random_test_images(model):
    classes = (
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    )
    
    fig, axes = plt.subplots(3, 3)
    for row in axes:
        for plot in row:
            index = random.randint(0, len(test))
            plot.imshow(test[index][0].permute(1, 2, 0))
            plot.set_xticks([])
            plot.set_yticks([])
            truth = classes[test[index][1]]
            pred = classes[torch.argmax(model(test[index][0])).item()]
            plot.set_title(f'{pred} ({truth})')

    plt.show()
    
# random_test_images(model)
