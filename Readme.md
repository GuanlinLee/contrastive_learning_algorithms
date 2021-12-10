#Contrastive Learning

This repo includes [MoCo v2](https://arxiv.org/abs/2003.04297),
[SimCLR](https://arxiv.org/abs/2002.05709), 
[SwAV](https://arxiv.org/abs/2006.09882) 
and [Barlow Twins](https://arxiv.org/abs/2103.03230). You can freely choose 
the encoder's architecture, e.g., ResNet, DenseNet, ResNeXt, VGG, WiderResNet and Mobilenet.

##Requirements
We tested on Pytorch-1.9.0. But it should work if you use Pytorch>=1.6.0.


##Datasets
We support Cifar-10, Cifar-100, STL-10 and Tiny-Imagenet.
You can freely add your dataset in [DataLoader.py](https://github.com/GuanlinLee/contrastive_learning_algorithms/blob/main/DataLoader.py).

##How to use
It is easy to train your encoder with our supported contrastive learning methods.
```
python train_encoder.py --method [moco, swav, barlowtwins, simclr] --arch [resnet18,resnet34,
resnet50, densenet121, mobilenetv2, vgg11, wrn28x10, resnext2x64d] --data_path [path you store the data]
--dataset [cifar10, cifar100, stl10, tiny_imagenet] --gpu 0 --batch_size 128 --epochs 200
```





##Acknowledgement
Most of our codes are borrowed from the listed repos:
+ https://github.com/facebookresearch/swav
+ https://github.com/facebookresearch/moco
+ https://github.com/facebookresearch/barlowtwins
+ https://github.com/IgorSusmelj/barlowtwins
+ https://github.com/abhinavagarwalla/swav-cifar10
+ https://github.com/leftthomas/SimCLR
