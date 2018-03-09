# pytorch_deephash

## Introduction

This is the Pytorch implementation of [Deep Learning of Binary Hash Codes for Fast Image Retrieval](https://github.com/kevinlin311tw/caffe-cvprw15), and can achieve more than 94% mAP in CIFAR10 dataset.

## Environment

> Pytorch 0.2.0_4

> torchvision 0.1.9

## Training

```python
python train.py
```

You will get trained models in model folder by default, and models' names are their test accuracy.

## Evaluation

```python
python mAP.py --pretrained {your saved model name in model folder by default}
```

## Tips

There are some other args, which you can get them by adding '-h' or reading the code.

## To do

Add the implementation of NUS-WIDE dataset
