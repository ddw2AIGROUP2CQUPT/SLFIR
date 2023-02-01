from time import time
from eval_model import SLFIR_Model
import time
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
hp.backbone_model_dir = './' + hp.dataset_name + '_' + str(hp.feature_num) + '_backbone.pth'
hp.attn_model_dir = './' + hp.dataset_name + '_' + str(hp.feature_num) + '_attn.pth'
hp.lstm_model_dir = './' + hp.dataset_name + '_' + str(hp.feature_num) + '_lstm.pth'
print(hp)

slfir_model = SLFIR_Model(hp)
dataloader_sketch_train, dataloader_sketch_test = get_dataloader(hp)

def fixed_network():
    slfir_model.backbone_network.load_state_dict(torch.load(hp.backbone_model_dir, map_location=device))
    slfir_model.backbone_network.to(device)
    slfir_model.backbone_network.fixed_param()
    slfir_model.backbone_network.eval()

    slfir_model.attn_network.load_state_dict(torch.load(hp.attn_model_dir, map_location=device))
    slfir_model.attn_network.to(device)
    slfir_model.attn_network.fixed_param()
    slfir_model.attn_network.eval()

    slfir_model.lstm_network.load_state_dict(torch.load(hp.lstm_model_dir, map_location=device))
    slfir_model.lstm_network.to(device)
    for param in slfir_model.lstm_network.parameters():
        param.requires_grad = False
    slfir_model.lstm_network.eval()

def main_eval():
    fixed_network()

    with torch.no_grad():
        start_time = time.time()
        top1, top5, top10, mean_IOU, mean_MA, mean_OurB, mean_OurA = slfir_model.evaluate_NN(dataloader_sketch_test)
        print("TEST A@1: {}".format(top1))
        print("TEST A@5: {}".format(top5))
        print("TEST A@10: {}".format(top10))
        print("TEST M@B: {}".format(mean_IOU))
        print("TEST M@A: {}".format(mean_MA))
        print("TEST OurB: {}".format(mean_OurB))
        print("TEST OurA: {}".format(mean_OurA))
        print("TEST Time: {}".format(time.time()-start_time))

if __name__ == "__main__":
    main_eval()




