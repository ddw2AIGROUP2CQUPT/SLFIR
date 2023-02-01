# Sketch Less Face Image Retrieval: A New Challenge

https://user-images.githubusercontent.com/107622162/198173969-ab970fbb-e4f2-4510-9314-b587b6bda99a.mp4

**If you want to watch a higher quality video, you can click [Demo Video](https://youtu.be/gZfRjrY5H0Y) to access youtube video or click [Demo Video](https://github.com/ddw2AIGROUP2CQUPT/SLFIR/blob/main/demo.mp4?raw=true) to download only 80M video to watch.**

**This repository is the official pytorch implementation of our paper, \*Sketch Less Face Image Retrieval: A New Challenge\*.**

## 🌟 Pipeline

![img](README.assets/wps1.png)

## :floppy_disk: Dataset

### Please click on the link [FS2K-SDE](https://github.com/ddw2AIGROUP2CQUPT/FS2K-SDE) for the dataset.

![image-20221025194710409](README.assets/image-20221025194710409.png)

## 📁Source Code

**The source code for this project is in the `src` folder of the repository.**  
**To facilitate anyone's reproduction and further study of our work, we provide the `full training code` and `test code` for stage1 and stage2 (only stage2 is included), the `trained model files` and the `log files` generated during the training process.**

### Train

#### Stage1

To train for stage1 simply enter the following command in the terminal:

```python
CUDA_VISIBLE_DEVICES=0 python train.py \ 
						--dataset_name Face-1000
						--root_dir {your_root_proj_path}
						--nTheads 4
						--backbone_lr 5e-4
						--lr 5e-3
						--max_epoch 200
						--feature_num 16
```

Or just edit the `parser` parameter at the bottom of the train.py file and run it the way you like.

**Parameters:**

```python
--dataset_name, type=str, default='Face-1000', help='Face-1000 / Face-450'
--root_dir, type=str, default='./', help='The root directory of the entire project file'
--nThreads, type=int, default=4
--backbone_lr, type=float, default=0.0005, help='Learning rate of the backbone network'
--lr, type=float, default=0.005, help='Learning rate of LSTM or MLP'
--max_epoch, type=int, default=200
--print_freq_iter, type=int, default=1, help='Step rate for printing debug messages'
--feature_num, type=int, default=16, help='Number of features in the last layer of the neural network'
```

#### Stage2

To train for stage2 simply enter the following command in the terminal:

```python
CUDA_VISIBLE_DEVICES=0 python train.py \ 
						--dataset_name Face-1000
						--root_dir {your_root_proj_path}
    						--batchsize 32
						--nTheads 4
						--lr 5e-4
						--max_epoch 300
						--feature_num 16
```

Or just edit the `parser` parameter at the top of the train.py file and run it the way you like.

**Parameters:**

```python
--dataset_name, type=str, default='Face-1000', help='Face-1000 / Face-450'
--root_dir, type=str, default='./', help='The root directory of the entire project file'
--batchsize, type=int, default=32
--nThreads, type=int, default=4
--lr, type=float, default=0.0005, help='Learning rate of LSTM or MLP'
--epoches, type=int, default=300
--feature_num, type=int, default=16, help='Number of features in the last layer of the neural network'
```

### Eval

We only provide the validation code for stage2 and it comes with the tensorboard records (in the run directory) and log log files from our training process.

To eval for stage2 simply enter the following command in the terminal:

```python
CUDA_VISIBLE_DEVICES=0 python train.py \ 
						--dataset_name Face-1000
						--root_dir {your_root_proj_path}
    						--batchsize 32
						--nTheads 4
						--lr 5e-4
						--max_epoch 300
						--feature_num 16
```

## ⏳ To Do

- [x] Release training code

- [x] Release testing code
- [x] Release pre-trained models

## 📔 Citation

coming soon......

## 💡 Acknowledgments

*We would like to thank all of reviewers for their constructive comments and CQUPT for supporting.*

## 📨 Contact

This repo is currently maintained by Dawei Dai (dw_dai@163.com) and his master's student Yutang Li (2018211556@stu.cqupt.edu.cn).
