"""
Date: 2021/05/10
Author: worith
"""

import torch
import argparse
import os
import time
import numpy as np
import pandas as pd
import copy

from model.model import NetX2Y, NetH2Y

import torch.utils.data as Data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from dataset.fracture_dataset import PDDataset
from config.config import global_config
from sklearn.model_selection import train_test_split

plt.rcParams['font.size'] = 18
plt.rcParams['font.sans-serif'] = 'Times New Roman'


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser(description='agent model of frature')

parser.add_argument('--trainer_name', default=global_config.getRaw('config', 'model_name'))
parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--h2y', action='store_true', help="Hidden to output")
parser.add_argument('--x2y', action='store_true', help="input to output")
parser.add_argument('--six_stages', action='store_true', help='2 stages data or 6 stages')


args = parser.parse_args()

# global config
data_path = global_config.getRaw('config', 'data_base_path')
stages = global_config.getRaw('config', 'stages')
runs_save_folder = os.path.join(global_config.getRaw('config', 'runs_save_folder'), stages)
model_save_folder = os.path.join(global_config.getRaw('config', 'model_save_folder'), stages)
best_h2y_model = global_config.getRaw('config', 'best_h2y_model')
best_x2y_added_model = global_config.getRaw('config', 'best_x2y_added_model')
best_x2y_model = global_config.getRaw('config', 'best_x2y_model')


if args.six_stages:
    model_save_folder = './checkpoint/six_stages'
    runs_save_folder = './runs/events/six_stages'


if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

# train config
epochs = int(global_config.getRaw('train', 'num_epochs'))
batch_size = int(global_config.getRaw('train', 'batch_size'))
base_lr = float(global_config.getRaw('train', 'lr'))
save_freq = int(global_config.getRaw('train', 'save_freq'))
random_seed = int(global_config.getRaw('train', 'random_seed'))
add_physical_info = int(global_config.getRaw('train', 'add_physical_info'))

# writer = SummaryWriter(os.path.join(runs_save_folder), '%s' % args.trainer_name)
model_dir = os.path.join(model_save_folder, '%s/' % args.trainer_name)


def main():
    # load data
    if args.six_stages:
        file_path = os.path.join(data_path, '6_stages.csv')
        data = pd.read_csv(file_path)
        dataset = PDDataset('6', file_path)
    else:
        file_path = os.path.join(data_path, 'fracture_20201210.csv')
        data = pd.read_csv(file_path)
        dataset = PDDataset('2', file_path)

    npv_data = np.array(data['NPV'])
    max_npv, min_npv = np.max(npv_data), np.min(npv_data)

    train_data, val_test_data = train_test_split(dataset, test_size=0.2, random_state=random_seed)

    test_data, val_data = train_test_split(val_test_data, test_size=0.5, random_state=random_seed)

    test_loader = Data.DataLoader(
        dataset=test_data,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )

    if args.six_stages:
        net_h2y = NetH2Y(80, 40, 20, len(dataset.hidden_feat), n_output=len(dataset.out_feat))
        net_x2y = NetX2Y(20, 40, 20, 20, add_physical_info, n_feature=len(dataset.in_feat),
                         n_output=len(dataset.out_feat))
    else:
        net_h2y = NetH2Y(20, 40, 20, len(dataset.hidden_feat), n_output=len(dataset.out_feat))
        net_x2y = NetX2Y(20, 40, 20, 20, add_physical_info, n_feature=len(dataset.in_feat), n_output=len(dataset.out_feat))

    loss_func = torch.nn.MSELoss()
    if args.h2y:
        test(net_h2y, net_x2y, loss_func, test_loader, add_physical_info,
             stage=1)

    if args.x2y:
        pred, target = test(net_h2y, net_x2y, loss_func, test_loader, add_physical_info,
                            stage=2)
        # 归一化还原
        target = [i * (max_npv - min_npv) + min_npv for i in target]
        pred = [i * (max_npv - min_npv) + min_npv for i in pred]
        pred_dict = {'NPV_pred': pred, 'NPV': target}
        pred_data = pd.DataFrame(pred_dict)
        pred_data.to_csv('data/%s_pred_NPV.csv' % stages, index=None)


def test(model_1, model_2, loss_func, loader, add_physical_info, stage):
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        losses = []
        pred, target = [], []
        epoch_start_time = time.time()
        text_name = ""
        for step, (x, h, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习

            if stage == 1:
                text_name = "h2y"
                model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_h2y_model))
                prediction, _ = model_1(h)
                loss = loss_func(prediction, y)
            elif stage == 2:
                if add_physical_info:
                    text_name = "x2y_added"
                    model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_h2y_model))
                    _, physical_info = model_1(h)
                    model_2.load_state_dict(torch.load(model_save_folder + "/%s" % best_x2y_added_model))
                    model_2.add_physical_info(physical_info)
                    prediction = model_2(x)
                else:
                    text_name = "x2y"
                    model_2.load_state_dict(torch.load(model_save_folder + "/%s" % best_x2y_model))
                    prediction = model_2(x)
                loss = loss_func(prediction, y)
            else:
                print("please input the correct stage")
                return

            losses.append(loss.data.item())
            if step == 0:
                pred = prediction.detach().numpy()
                target = y.detach().numpy()
            else:
                pred = np.concatenate((np.array(pred), prediction.detach().numpy()), 0)
                target = np.concatenate((np.array(target), y.detach().numpy()), 0)
            # print(
            #     f"Stage: {stage}\t Epoch: {epoch} \t Batch_num: {step} \t Loss={loss.data.cpu():.4} \t "
            #     f"Time={time.time() - start_time:.4}")

        error = np.linalg.norm(target - pred, 2) / np.linalg.norm(target, 2)

        print(f"Test of {text_name}: AVG_Loss={np.mean(losses):.4} \t Time={time.time() - epoch_start_time:.4} \t"
              f" l2_error={error:.4}")
    return pred, target


if __name__ == '__main__':
   main()