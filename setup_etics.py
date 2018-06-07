import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import pyro
import bcolz


def main():

    print('Processing VGG16 model ... ')
    model_vgg = torchvision.models.vgg16(pretrained=True)

    print('Processing ResNet18 model ... ')
    model_resnet = torchvision.models.resnet18(pretrained=True)

    print('Processing CIFAR10 dataset ... ')
    trainset_cifar = torchvision.datasets.CIFAR10(root='./data', download=True)

    print('Processing MNIST dataset ... ')
    trainset_mnist = torchvision.datasets.MNIST(root='./data', download=True)

    print("That's all folks!")


if __name__ == '__main__':

    main()    