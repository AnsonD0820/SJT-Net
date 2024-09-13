# **Spatial Reconstructed Joint Training Transformer Network for Cross-Domain Remote Sensing Images Semantic Segmentation** <br />

This is a pytorch implementation of SJT-Net.

<div align="center">
  <img src="https://github.com/AnsonD0820/SJT-Net/blob/main/Fig1.png">
</div> <br />

## Prerequisites <br />
* Python 3.9
* Pytorch 2.0.0
* torchvision 0.15.0
* OpenCV
* numpy
* yacs
* CUDA >= 11.7
<br />

## Train
### Source doamin segmentation training stage
python train_source_seg.py

### Adversarial training stage
python train_adversarial_training.py

### Target domain self-training stage
python train_self_training.py
