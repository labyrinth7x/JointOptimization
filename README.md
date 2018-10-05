# Joint Optimization Framework for Learning with Noisy Labels
This is an implmentation for the paper [Joint Optimization Framework for Learning with Nosiy Labels](https://arxiv.org/pdf/1803.11364.pdf)/br>
The official implementation is [Chainer](https://github.com/DaikiTanaka-UT/JointOptimization)<br>
## Requirement<br>
* Python 3.5 
* Pytorch 0.4 & torchvision
* numpy
* matplotlib (not necessary unless the need for the result figure)  
## Network<br>
The backbone of the network is Resnet-34.<br>
It is implemented in /utils/renset.py.<br>
Here only Resnet-34 and Resnet-18 is available.<br>
## Train  
There are two steps to folllow.  
* First: Train the noisy dataset and update labels.
Train the network on the Symmetric Noise CIFAR-10 dataset:
```
python train.py --out "./model_data/" --gpus 0 --noise_ratio 0.2 --alpha 0.8 --beta 0.4 --dataset_type "sym_noise"  
```
Train the network on the Asymmetric Noise CIFAR-10 dataset: 
```
python train.py --out "./model_data/" --gpus 0 --noise_ratio 0.2 --alpha 0.8 --beta 0.4 --dataset_type "asym_noise" 
```
* Second: Retrain the updated dataset.  
Train the network on the Symmetric Noise CIFAR-10 dataset:  
```
python retrain.py --gpus 0 --out "./model_data/" --gpus 0 --noise_ratio 0.2 --alpha 0.8 --beta 0.4 --dataset_type "sym_noise"  
```
Train the network on the Asymmetric Noise CIFAR-10 dataset:  
```
python retrain.py --gpus 0 --out "./model_data/" --gpus 0 --noise_ratio 0.2 --alpha 0.8 --beta 0.4 --dataset_type "asym_noise"  
```
## References
* D. Tanaka, D. Ikami, T. Yamasaki and K. Aizawa. "Joint Optimization Framework for Learning with Noisy Labels", in CVPR, 2018.
* Another unofficial implementation for the same paper. [here](https://github.com/YU1ut/JointOptimization)
