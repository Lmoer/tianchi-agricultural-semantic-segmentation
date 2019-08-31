# Project
## 依赖项
- *Python3.6*
- *Pytorch 1.1.0*
- torchvision0.3.0
- *cv2*
- *numpy*
- *PIL*
- CUDA Version: 10.1 
## 安装依赖项
- 确认安装Anaconda
- 创建环境并且激活环境，注意激活project环境后不要退出

```
conda create -n project python=3.6
```

```
source activate project
```


- 安装pytorch 1.1.0

```
pip install torch==1.1.0或pip install http://mirrors.aliyun.com/pypi/packages/69/60/f685fb2cfb3088736bafbc9bdbb455327bdc8906b606da9c9a81bae1c81e/torch-1.1.0-cp36-cp36m-manylinux1_x86_64.whl
```

```
pip install torchvision==0.3.0或pip install http://mirrors.aliyun.com/pypi/packages/2e/45/0f2f3062c92d9cf1d5d7eabd3cae88cea9affbd2b17fb1c043627838cb0a/torchvision-0.3.0-cp36-cp36m-manylinux1_x86_64.whl
```

- 安装PIL库

```
pip install pillow
```

- 安装cv2

```
pip install opencv-python
```
## 数据准备

提前解压image_1.png、image_2.png、image_3.png、image_4.png、image_1_label.png、 image_2_label.png放在data目录下。数据切割比较耗时，需要3个小时左右的时间，目录如下。

|–data

​	|-- model	

​	|-- train

​		|--data

​		|--label

​	|-- test

​		|--image3

​		|--mask3

​		|--image4

​		|--mask4

​	|-- jingwei_round1_train_20190619/image_1.png

​	|-- jingwei_round1_train_20190619/image_2.png

​	|-- jingwei_round1_train_20190619/image_1_label.png

​	|-- jingwei_round1_train_20190619/image_2_label.png

​	|--jingwei_round1_test_a_20190619/image_3.png

​	|-- jingwei_round1_test_a_20190619/image_4.png



```
python generate_train_data.py
python generate_test_data.py
```
## 模型训练
（1）使用deeplabv3,交叉熵训练。优化器SGD

```
python train.py --train --exp bn_lr7e-3 --epochs 20 --base_lr 0.007
```
（2）使用deeplabv3,focalloss和lovasz_losses训练。优化器SGD

```
python train_focalloss.py --train --exp bn_lr7e-3 --epochs 20 --base_lr 0.007
```

focalloss，在训练时可以根据样本的学习难易程度调整样本的loss权重，使模型可以更加关注难学习的样本。

lovasz_losses ：语义分割一般用IOU去衡量模型的表现，但IOU不是一个可导函数，利用IOU直接训练模型会导致训练过程的不稳定。一个模型从坏到好，我们希望监督它的loss/metric的过渡是平滑的，直接暴力套用IoU显然不行，怎么优化IOU一直是语义分割要面临的问题，一般的baseline论文一般通过优化cross entropy去优化IoU。但优化cross entropy并不等同于优化IoU。LovaszSoftmax可以作为IOU的替代目标函数。

github链接https://github.com/bermanmaxim/LovaszSoftmax

（3）使用deeplabv3,focalloss和lovasz_losses训练。基于余弦退火的SWA（随机梯度平均）优化器

基于余弦退火的SWA（随机梯度平均）优化器[https://github.com/Lmoer/learn_blog/blob/master/lavz_loss%E4%BB%A5%E5%8F%8Aswa%E7%AC%94%E8%AE%B0.md](https://github.com/Lmoer/learn_blog/blob/master/lavz_loss以及swa笔记.md)

```
python train_swa.py --train --exp bn_lr7e-3 --epochs 30 --base_lr 0.007
```

（4）使用deeplabv3,focalloss和lovasz_losses训练。512 * 512, 448 * 448 多尺度交替训练。增强模型不同尺度的检测准确率

```
python train_multiscale.py --train --exp bn_lr7e-3 --epochs 30 --base_lr 0.007
```

（5）模型融合，投票。（使用backbone为resnet50，引入attention机制 的Unet 训练模型，以及deeplabv3模型的预测结果进行融合（文件夹Unet_resnet34/））

```
python vote.py
```

## 模型预测及预测结果拼接

```
python predict.py
```

图像后处理，填充空洞

```
python fill_hole.py
```
