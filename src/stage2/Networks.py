import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
import torch
from torch.autograd import Variable
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class InceptionV3_Network(nn.Module):
    def __init__(self):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e
        # # 固定前面层的参数
        # for param in self.parameters():
        #     param.requires_grad = False
        # # 后面这些层仍然使用预训练的参数，但用小学习率更新
        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
            

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        return F.normalize(x)
    
    def fixed_param(self):
        for param in self.parameters():
            param.requires_grad = False


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=1),
                            nn.BatchNorm2d(512),
                            nn.ReLU(),
                            nn.Conv2d(512, 1, kernel_size=1))
        self.pool_method = nn.AdaptiveMaxPool2d(1)  # as default
    
    def forward(self, x):
        attn_mask = self.net(x)
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3))
        x = x + (x * attn_mask)
        x = self.pool_method(x).view(-1, 2048)
        return F.normalize(x)
    
    def fixed_param(self):
        for param in self.parameters():
            param.requires_grad = False


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.head_layer = nn.Linear(2048, 16)
    
    def forward(self, x):
        return F.normalize(self.head_layer(x))


class Block_lstm(nn.Module):
    def __init__(self, opt):
        super(Block_lstm, self).__init__()
        self.opt = opt
        self.lstm_0 = nn.LSTM(input_size=2048, hidden_size=512, bidirectional=True)
        self.lstm_1 = nn.LSTM(input_size=1024, hidden_size=int(self.opt.feature_num // 2), bidirectional=True)
    def attention_net(self, lstm_output):
        sequence_length, batch_size, hidden_lay = lstm_output.shape
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, hidden_lay])
        w_omega = Variable(torch.zeros(self.opt.feature_num, batch_size).to(self.opt.device))
        u_omega = Variable(torch.zeros(batch_size).to(self.opt.device))
        attn_tanh = torch.tanh(torch.mm(output_reshape, w_omega))
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, sequence_length, 1])
        state = lstm_output.permute(1, 0, 2)
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output
    def forward(self, X):

        X = X.unsqueeze(dim=0)
        _,b,_ = X.shape
        hidden_state = torch.zeros(1*2, b, 512)
        cell_state = torch.zeros(1*2, b, 512)
        outputs, (_, _) = self.lstm_0(X, (hidden_state.to(self.opt.device), cell_state.to(self.opt.device)))
        hidden_state_1 = torch.zeros(1 * 2, b, int(self.opt.feature_num // 2))
        cell_state_1 = torch.zeros(1 * 2, b, int(self.opt.feature_num // 2))
        outputs, (_, _) = self.lstm_1(outputs, (hidden_state_1.to(self.opt.device), cell_state_1.to(self.opt.device)))
        outputs = self.attention_net(outputs)
        # outputs = outputs.squeeze(dim=0)
        return outputs
