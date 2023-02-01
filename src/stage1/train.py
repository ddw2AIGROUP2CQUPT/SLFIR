import torch
import time
from model import SLFIR_Model
from dataset import get_dataloader

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
import argparse

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SLFIR Model')
    parser.add_argument('--dataset_name', type=str, default='Face-1000', help='Face-1000 / Face-450')
    parser.add_argument('--root_dir', type=str, default='/home/ubuntu/lxd-workplace/LYT/face-sbir-github')
    parser.add_argument('--nThreads', type=int, default=4)
    parser.add_argument('--backbone_lr', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--print_freq_iter', type=int, default=1)
    parser.add_argument('--feature_num', type=int, default=16)
    hp = parser.parse_args()
    if hp.dataset_name == 'Face-1000':
        hp.batchsize = 64
        hp.eval_freq_iter = 50
    elif hp.dataset_name == 'Face-450':
        hp.batchsize = 32
        hp.eval_freq_iter = 20

    dataloader_Train, dataloader_Test = get_dataloader(hp)
    # 返回的是草图训练集和测试题
    print(hp)

    model = SLFIR_Model(hp)
    model.to(device)
    step_count, top1, top5, top10, top50, top100 = -1, 0, 0, 0, 0, 0

    for i_epoch in range(hp.max_epoch):
        for batch_data in dataloader_Train:
            step_count = step_count + 1
            start = time.time()
            model.train()
            loss = model.train_model(batch=batch_data)

            if step_count % hp.print_freq_iter == 0:
                print('Epoch: {}, Iteration: {}, Loss: {:.8f}, Top1_Accuracy: {:.5f}, Top5_Accuracy; {:.5f}, Top10_Accuracy: {:.5f}, Time: {}'.format(
                    i_epoch, step_count, loss, top1, top5, top10, time.time() - start))

            if step_count % hp.eval_freq_iter == 0 and i_epoch >= 20:
                with torch.no_grad():
                    top1_eval, top5_eval, top10_eval, top50_eval, top100_eval = model.evaluate(dataloader_Test)
                    print('results : ', top1_eval, ' / ', top5_eval, ' / ', top10_eval, ' / ', top50_eval, ' / ', top100_eval)
                # model update

                if top10_eval > top10:
                    torch.save(model.backbone_network.state_dict(),
                               hp.dataset_name + '_' + str(hp.feature_num) + '_backbone_best.pth')
                    torch.save(model.attn_network.state_dict(),
                               hp.dataset_name + '_' + str(hp.feature_num) + '_attn_best.pth')
                    torch.save(model.linear_network.state_dict(),
                               hp.dataset_name + '_' + str(hp.feature_num) + '_linear_best.pth')
                    top1, top5, top10, top50, top100 = top1_eval, top5_eval, top10_eval, top50_eval, top100_eval
                    print('Model Updated')
