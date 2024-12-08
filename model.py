import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # Output corresponds to num_classes
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = ResNet18()
    print(model)

class VGG11Model(nn.Module):
    # Implement VGG11 model for transfer learning
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg11(pretrained=True)
        
        # Freeze the convolutional base
        # for param in self.model.features.parameters():
        #     param.requires_grad = False
        
        # Replace avgpool with AdaptiveAvgPool2d
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Replace the classifier with a new one
        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # Output corresponds to num_classes
        )

    def forward(self, x):
        return self.model(x)