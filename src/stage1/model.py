import torch.nn as nn
from Networks import InceptionV3_Network, Attention, Linear
from torch import optim
import torch
import time
import torch.nn.functional as F

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class SLFIR_Model(nn.Module):
    def __init__(self, hp):
        super(SLFIR_Model, self).__init__()
        # inception预训练网络
        self.backbone_network = InceptionV3_Network()
        self.backbone_train_params = self.backbone_network.parameters()
        # 定义权重初始化方法
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)
        # 注意力块
        self.attn_network = Attention()
        self.attn_network.apply(init_weights)
        self.attn_train_params = self.attn_network.parameters()
        # 线性层
        self.linear_network = Linear(hp.feature_num)
        self.linear_network.apply(init_weights)
        self.linear_train_params = self.linear_network.parameters()

        self.optimizer = optim.Adam([
            {'params': filter(lambda param: param.requires_grad, self.backbone_train_params), 'lr': hp.backbone_lr},
            {'params': self.attn_train_params, 'lr': hp.lr},
            {'params': self.linear_train_params, 'lr': hp.lr}])
        # 训练的模型
        self.loss = nn.TripletMarginLoss(margin=0.3)
        self.hp = hp
    
    def train_model(self, batch):
        self.train()
        positive_feature = self.linear_network(self.attn_network(self.backbone_network(batch['positive_img'].to(device))))
        negative_feature = self.linear_network(self.attn_network(self.backbone_network(batch['negative_img'].to(device))))
        sample_feature = self.linear_network(self.attn_network(self.backbone_network(batch['sketch_img'].to(device))))

        loss = self.loss(sample_feature, positive_feature, negative_feature)
        # for opt in self.optimizer:
        #     opt.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # for opt in self.optimizer:
        #     opt.step()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, datloader_Test):
        Image_Feature_ALL = []
        # Image_Name = []
        Sketch_Feature_ALL = []
        # Sketch_Name = []
        start_time = time.time()
        self.eval()
        for _, sanpled_batch in enumerate(datloader_Test):
            sketch_feature, positive_feature = self.test_forward(sanpled_batch)
            Sketch_Feature_ALL.extend(sketch_feature)
            Image_Feature_ALL.extend(positive_feature)

        rank = torch.zeros(len(Sketch_Feature_ALL))
        Image_Feature_ALL = torch.stack(Image_Feature_ALL)

        for num, sketch_feature in enumerate(Sketch_Feature_ALL):
            distance = F.pairwise_distance(sketch_feature.unsqueeze(0), Image_Feature_ALL)
            target_distance = F.pairwise_distance(sketch_feature.unsqueeze(0),
                                                  Image_Feature_ALL[num].unsqueeze(0))

            rank[num] = distance.le(target_distance).sum()
        top1 = rank.le(1).sum().numpy() / rank.shape[0]
        top5 = rank.le(5).sum().numpy() / rank.shape[0]
        top10 = rank.le(10).sum().numpy() / rank.shape[0]
        top50 = rank.le(50).sum().numpy() / rank.shape[0]
        top100 = rank.le(100).sum().numpy() / rank.shape[0]

        print('Time to EValuate:{}'.format(time.time() - start_time))
        return top1, top5, top10, top50, top100

    def test_forward(self, batch):  # this is being called only during evaluation
        sketch_feature = self.linear_network(self.attn_network(self.backbone_network(batch['sketch_img'].to(device))))
        positive_feature = self.linear_network(self.attn_network(self.backbone_network(batch['positive_img'].to(device))))
        return sketch_feature, positive_feature
