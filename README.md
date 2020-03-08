# pytorch_deephash

## Introduction

This is the Pytorch implementation of [Deep Learning of Binary Hash Codes for Fast Image Retrieval](https://github.com/kevinlin311tw/caffe-cvprw15), and can achieve more than 93% mAP in CIFAR10 dataset.

## Environment

> Pytorch 1.4.0
>
> torchvision 0.5.0
>
> tqdm
>
> numpy


## Training

```bash
python train.py
```

You will get trained models in model folder by default, and models' names are their test accuracy.

## Evaluation

```bash
python evaluate.py --pretrained {your saved model name in model folder by default}
```

## Tips

1. If using Windows, keep num_works zero

2. There are some other args, which you can get them by adding '-h' or reading the code.
