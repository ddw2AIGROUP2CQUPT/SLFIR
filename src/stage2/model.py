import torch.nn as nn
from Networks import InceptionV3_Network, Attention, Block_lstm
from torch import optim
import torch
import numpy as np
import torch.nn.functional as F

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SLFIR_Model(nn.Module):
    def __init__(self, opt):
        super(SLFIR_Model, self).__init__()
        # inception预训练网络
        self.backbone_network = InceptionV3_Network()
        self.backbone_network.load_state_dict(torch.load(opt.backbone_model_dir, map_location=opt.device))
        self.backbone_network.to(opt.device)
        self.backbone_network.fixed_param()
        self.backbone_network.eval()
        # self.backbone_train_params = self.backbone_network.parameters()
        # 定义权重初始化方法
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        # 注意力块
        self.attn_network = Attention()
        self.attn_network.load_state_dict(torch.load(opt.attn_model_dir, map_location=opt.device))
        self.attn_network.to(opt.device)
        self.attn_network.fixed_param()
        self.attn_network.eval()
        # self.attn_train_params = self.attn_network.parameters()
        # lstm网络
        self.lstm_network = Block_lstm(opt)
        self.lstm_network.apply(init_weights)
        self.lstm_network.train()
        self.lstm_network.to(opt.device)
        self.lstm_train_params = self.lstm_network.parameters()

        self.optimizer = optim.Adam([
            {'params': self.lstm_train_params, 'lr': opt.lr}])

        # 训练的模型
        self.loss = nn.TripletMarginLoss(margin=0.3, p=2)
        self.opt = opt
    
    def train_model(self, batch):
        self.backbone_network.eval()
        self.attn_network.eval()
        self.lstm_network.train()
        loss = 0
        for idx in range(len(batch['sketch_seq'])):
            sketch_seq_feature = self.lstm_network(self.attn_network(
                self.backbone_network(batch['sketch_seq'][idx].to(self.opt.device))))
            positive_feature = self.lstm_network(self.attn_network(
                self.backbone_network(batch['positive_img'][idx].unsqueeze(0).to(self.opt.device))))
            negative_feature = self.lstm_network(self.attn_network(
                self.backbone_network(batch['negative_img'][idx].unsqueeze(0).to(self.opt.device))))
            positive_feature = positive_feature.repeat(sketch_seq_feature.shape[0], 1)
            negative_feature = negative_feature.repeat(sketch_seq_feature.shape[0], 1)
            loss += self.loss(sketch_seq_feature, positive_feature, negative_feature)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_NN(self, dataloader):
        self.backbone_network.eval()
        self.attn_network.eval()
        self.lstm_network.eval()

        self.Sketch_Array_Test = []
        self.Image_Array_Test = []
        for idx, batch in enumerate(dataloader):
            sketch_feature = self.attn_network(
                self.backbone_network(batch['sketch_seq'].squeeze(0).to(self.opt.device)))
            positive_feature = self.lstm_network(self.attn_network(
                self.backbone_network(batch['positive_img'].to(self.opt.device))))
            self.Sketch_Array_Test.append(sketch_feature)
            self.Image_Array_Test.append(positive_feature)
        self.Sketch_Array_Test = torch.stack(self.Sketch_Array_Test)
        self.Image_Array_Test = torch.stack(self.Image_Array_Test).view(self.Sketch_Array_Test.shape[0], -1)
        num_of_Sketch_Step = len(self.Sketch_Array_Test[0])
        avererage_area = []
        avererage_area_percentile = []
        avererage_ourB = []
        avererage_ourA = []
        exps = np.linspace(1, num_of_Sketch_Step, num_of_Sketch_Step) / num_of_Sketch_Step
        factor = np.exp(1 - exps) / np.e
        rank_all = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        rank_all_percentile = torch.zeros(len(self.Sketch_Array_Test), num_of_Sketch_Step)
        # sketch_range = torch.Tensor(sketch_range)
        for i_batch, sanpled_batch in enumerate(self.Sketch_Array_Test):
            mean_rank = []
            mean_rank_percentile = []
            mean_rank_ourB = []
            mean_rank_ourA = []
            for i_sketch in range(sanpled_batch.shape[0]):
                sketch_feature = self.lstm_network(sanpled_batch[:i_sketch+1].to(self.opt.device))
                target_distance = F.pairwise_distance(F.normalize(sketch_feature[-1].unsqueeze(0).to(self.opt.device)), self.Image_Array_Test[i_batch].unsqueeze(0).to(self.opt.device))
                distance = F.pairwise_distance(F.normalize(sketch_feature[-1].unsqueeze(0).to(self.opt.device)), self.Image_Array_Test.to(self.opt.device))
                #rankingList = self.SortNameByData(distance, self.Image_Name_Test)
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                #a.le(b),，若a<=b，返回1
                #.sum， 算出来直接等于rank
                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                #(len-rank)/(len-1)
                if rank_all[i_batch, i_sketch].item() == 0:
                    #并不存在sum=0的情况，无用？
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    #1/(rank)
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                    mean_rank_ourB.append(1/rank_all[i_batch, i_sketch].item()*factor[i_sketch])
                    mean_rank_ourA.append(rank_all_percentile[i_batch, i_sketch].item()*factor[i_sketch])
                    #rank_percentile
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))

        print(rank_all)
        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        #A@1 A@5 A%10
        meanMB = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)

        return top1_accuracy, top5_accuracy, top10_accuracy, meanMB, meanMA, meanOurB, meanOurA

    def SortNameByData(self, dataList, nameList):
        convertDic = {}
        sortedDic = {}
        sortedNameList = []
        for index in range(len(dataList)):
            convertDic[index] = dataList[index]
        sortedDic = sorted(convertDic.items(), key=lambda item: item[1], reverse=False)
        for key, _ in sortedDic:
            sortedNameList.append(nameList[key])
        return sortedNameList

