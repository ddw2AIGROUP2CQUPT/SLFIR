from time import time
from model import SLFIR_Model
import time
import os
import torch
import numpy as np
import argparse
from dataset import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Face-1000', help='Face-1000 / Face-450')
parser.add_argument('--root_dir', type=str, default='../')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--nThreads', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--epoches', type=int, default=300)
parser.add_argument('--feature_num', type=int, default=16)
hp = parser.parse_args()
hp.device = device
if hp.dataset_name == 'Face-1000':
    hp.print_freq_iter = 50
    hp.lr = 0.0005
elif hp.dataset_name == 'Face-450':
    hp.print_freq_iter = 20
    hp.lr = 0.00005
hp.backbone_model_dir = '../stage1/' + hp.dataset_name + '_' + str(hp.feature_num) + '_backbone_best.pth'
hp.attn_model_dir = '../stage1/' + hp.dataset_name + '_' + str(hp.feature_num) + '_attn_best.pth'

print(hp)

tb_logdir = r"./run/"
slfir_model = SLFIR_Model(hp)
dataloader_sketch_train, dataloader_sketch_test = get_dataloader(hp)

def main_train():
    meanMB_buffer = 0
    real_p = [0, 0, 0, 0, 0, 0]
    loss_buffer = []
    tb_writer = SummaryWriter(log_dir=tb_logdir)
    Top1_Song = [0]
    Top5_Song = [0]
    Top10_Song = [0]
    meanMB_Song = []
    meanMA_Song = []
    meanWMB_Song = []
    meanWMA_Song = []
    step_stddev = 0

    for epoch in range(hp.epoches):
        for i, sanpled_batch in enumerate(dataloader_sketch_train):
            start_time = time.time()
            loss_triplet = slfir_model.train_model(sanpled_batch)
            loss_buffer.append(loss_triplet)
            # 累加损失
            step_stddev += 1
            tb_writer.add_scalar('total loss', loss_triplet, step_stddev)
            print('epoch: {}, iter: {}, loss: {}, time cost{}'.format(epoch, step_stddev, loss_triplet, time.time()-start_time))

            # 模型预热20个epoch，然后开始隔几个batchsize测试
            if epoch >= 20 and step_stddev % hp.print_freq_iter==0: #[evaluate after every 32*4 images]
                with torch.no_grad():
                    start_time = time.time()
                    top1, top5, top10, meanMB, meanMA, meanWMB, meanWMA = slfir_model.evaluate_NN(dataloader_sketch_test)
                    slfir_model.train()
                    print('Epoch: {}, Iteration: {}:'.format(epoch, step_stddev))
                    print("TEST A@1: {}".format(top1))
                    print("TEST A@5: {}".format(top5))
                    print("TEST A@10: {}".format(top10))
                    print("TEST M@B: {}".format(meanMB))
                    print("TEST M@A: {}".format(meanMA))
                    print("TEST W@MB: {}".format(meanWMB))
                    print("TEST W@MA: {}".format(meanWMA))
                    print("TEST Time: {}".format(time.time()-start_time))
                    Top1_Song.append(top1)
                    Top5_Song.append(top5)
                    Top10_Song.append(top10)
                    meanMB_Song.append(meanMB)
                    meanMA_Song.append(meanMA)
                    meanWMB_Song.append(meanWMB)
                    meanWMA_Song.append(meanWMA)
                    tb_writer.add_scalar('TEST A@1', top1, step_stddev)
                    tb_writer.add_scalar('TEST A@5', top5, step_stddev)
                    tb_writer.add_scalar('TEST A@10', top10, step_stddev)
                    tb_writer.add_scalar('TEST M@B', meanMB, step_stddev)
                    tb_writer.add_scalar('TEST M@A', meanMA, step_stddev)
                    tb_writer.add_scalar('TEST W@MB', meanWMB, step_stddev)
                    tb_writer.add_scalar('TEST W@MA', meanWMA, step_stddev)

                if meanMB > meanMB_buffer:
                    torch.save(slfir_model.backbone_network.state_dict(), hp.dataset_name + '_' + str(hp.feature_num) + '_backbone' + '.pth')
                    torch.save(slfir_model.attn_network.state_dict(), hp.dataset_name + '_' + str(hp.feature_num) + '_attn' + '.pth')
                    torch.save(slfir_model.lstm_network.state_dict(), hp.dataset_name + '_' + str(hp.feature_num) + '_lstm' + '.pth')
                    meanMB_buffer = meanMB

                    # # 这种做法会导致其他指标偏高
                    real_p = [top1, top5, top10, meanMA, meanWMB, meanWMA]
                    # 更改后符合保存模型时的真实指标
                    print('Model Updated')
                print('REAL performance: Top1: {}, Top5: {}, Top10: {}, MB: {}, MA: {}, WMB: {}, WMA: {}'.format(real_p[0], real_p[1],
                                                                                                                real_p[2],
                                                                                                                meanMB_buffer,
                                                                                                                real_p[3],
                                                                                                                real_p[4],
                                                                                                                real_p[5]))


    print("TOP1_MAX: {}".format(max(Top1_Song)))
    print("TOP5_MAX: {}".format(max(Top5_Song)))
    print("TOP10_MAX: {}".format(max(Top10_Song)))
    print("meaIOU_MAX: {}".format(max((meanMB_Song))))
    print("meaMA_MAX: {}".format(max((meanMA_Song))))
    print("meanWMB_MAX: {}".format(max(meanWMB_Song)))
    print("meanWMA_MAX: {}".format(max(meanWMA_Song)))
    print(Top1_Song)
    print(Top5_Song)
    print(Top10_Song)
    print(meanMB_Song)
    print(meanMA_Song)
    print(meanWMB_Song)
    print(meanWMA_Song)


if __name__ == "__main__":
    main_train()




