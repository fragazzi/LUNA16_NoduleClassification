import torch
import torch.nn as nn
import torchvision.models as models


# VGG16 or VGG19
class VGGExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        if model_name == 'VGG16':
            net = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        elif model_name == 'VGG19':
            net = models.vgg19(weights=models.VGG19_Weights.DEFAULT)

        # Use FC6 as output of the network (features)
        for i in range(len(net.classifier)):
            if i > 1:
                net.classifier[i] = nn.Identity()
        self.net = net
        
    def forward(self, x):
        x = self.net(x)
        
        return x
    

# AlexNet
class AlexNetExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

        # Use FC6 as output of the network (features)
        for i in range(len(net.classifier)):
            if i > 2:
                net.classifier[i] = nn.Identity()
                            
        self.net = net
        
    def forward(self, x):
        x = self.net(x)       
        return x
    

# ResNet
class ResNetExtractor(nn.Module):
    def __init__(self, depth):
        super().__init__()
        if depth == '18':
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif depth == '50':
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif depth == '101':
            net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif depth == '152':
            net = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            
        net.fc = nn.Identity()
        self.net = net
        
    def forward(self, x):
        x = self.net(x)       
        return x