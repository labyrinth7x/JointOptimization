import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    '''A re-implementation of BasicBlock.
    Only conv1 change the output size. (stride >= 1)
    '''
    expansion = 1

    def __init__(self, int_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(int_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # the base class nn.Module implements __call__()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    '''Only conv2 change the output size. (stride >= 1)'''
    expansion = 4

    def __init__(self, int_planes, out_planes, stride=1, downsample = None):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(int_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    '''Works well just for resnet18 & resnet34.'''
    def __init__(self, block, layers, num_classes=10):
        super(Resnet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)


    def _make_layer(self, block, out_planes, num_blocks, stride=1):
        downsample = None
        # residual should conv for the size consistency
        if (stride != 1) or (self.in_planes != out_planes):
            downsample = nn.Sequential(nn.Conv2d(self.in_planes, out_planes, 3, stride=stride, padding=1, bias=False),nn.BatchNorm2d(out_planes))

        layers = []
        layers.append(block(self.in_planes, out_planes, stride, downsample))
        for i in range(1, num_blocks):
            layers.append(block(out_planes, out_planes))

        self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out




def resnet18():
    return Resnet(BasicBlock, [2,2,2,2])

def resnet34():
    return Resnet(BasicBlock, [3,4,6,3])

'''
def resnet50():
    return Resnet(Bottleneck, [3,4,6,3])

def resnet101():
    return Resnet(Bottleneck, [3,4,23,3])

def resnet152():
    return Resnet(Bottleneck, [3,8,36,3])
'''