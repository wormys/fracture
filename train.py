"""
Date: 2021/05/10
Author: worith
"""

import torch
import argparse
import os
import time
import numpy as np
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
    lr = base_lr * (0.1 ** (epoch // 25))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser(description='agent model of frature')

parser.add_argument('--trainer_name', default=global_config.getRaw('config', 'model_name'))
parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--stage_1', action='store_true', help="Just test and terminate.")
parser.add_argument('--stage_2', action='store_true', help="Just test and terminate.")


args = parser.parse_args()

# global config
data_path = global_config.getRaw('config', 'data_base_path')
runs_save_folder = global_config.getRaw('config', 'runs_save_folder')
model_save_folder = global_config.getRaw('config', 'model_save_folder')
best_stage_1_model = global_config.getRaw('config', 'best_stage_1_model')

if not os.path.exists(model_save_folder):
    os.makedirs(model_save_folder)

# train config
epochs = int(global_config.getRaw('train', 'num_epochs'))
batch_size = int(global_config.getRaw('train', 'batch_size'))
base_lr = float(global_config.getRaw('train', 'lr'))
save_freq = int(global_config.getRaw('train', 'save_freq'))
random_seed = int(global_config.getRaw('train', 'random_seed'))
add_physical_info = int(global_config.getRaw('train', 'add_physical_info'))

writer = SummaryWriter(os.path.join(runs_save_folder), '%s' % args.trainer_name)
model_dir = os.path.join(model_save_folder, '%s/' % args.trainer_name)


def main():
    # load data
    file_path = os.path.join(data_path, 'fracture_20201210.csv')

    dataset = PDDataset(file_path)

    train_data, val_test_data = train_test_split(dataset, test_size=0.2, random_state=random_seed)

    test_data, val_data = train_test_split(val_test_data, test_size=0.5, random_state=random_seed)

    train_loader = Data.DataLoader(
        dataset=train_data,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )

    val_loader = Data.DataLoader(
        dataset=val_data,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )

    net_h2y = NetH2Y(len(dataset.hidden_feat), n_output=len(dataset.out_feat))
    net_x2y = NetX2Y(add_physical_info, n_feature=len(dataset.in_feat), n_output=len(dataset.out_feat))

    loss_func = torch.nn.MSELoss()
    if args.stage_1:
        optimizer_h2y = torch.optim.Adam(net_h2y.parameters(), lr=base_lr)
        train_writer_h2y = SummaryWriter(os.path.join(runs_save_folder, args.trainer_name + '_h2y'))

        best_h2y_error = np.inf
        best_h2y_model = copy.deepcopy(net_h2y.state_dict())
        for epoch in range(epochs):
            adjust_learning_rate(optimizer_h2y, epoch)
            train(net_h2y, net_x2y, optimizer_h2y, loss_func, train_writer_h2y, train_loader, add_physical_info,
                  epoch, stage=1)
            h2y_error = val(net_h2y, net_x2y, loss_func, train_writer_h2y, val_loader, add_physical_info,
                            epoch, stage=1)
            if h2y_error < best_h2y_error:
                best_h2y_error = h2y_error
                best_h2y_model = copy.deepcopy(net_h2y.state_dict())

        torch.save(best_h2y_model, model_save_folder + "/%s_best_model_h2y.pth" % args.trainer_name)

    if args.stage_2:
        if add_physical_info:

            train_writer_x2y = SummaryWriter(os.path.join(runs_save_folder, args.trainer_name + '_x2y_added'))
            save_x2y_model_path = model_save_folder + "/%s_best_model_x2y_added.pth" % args.trainer_name
        else:

            train_writer_x2y = SummaryWriter(os.path.join(runs_save_folder, args.trainer_name + '_x2y'))
            save_x2y_model_path = model_save_folder + "/%s_best_model_x2y.pth" % args.trainer_name
        optimizer_x2y = torch.optim.Adam(net_x2y.parameters(), lr=base_lr)

        best_x2y_error = np.inf
        best_x2y_model = copy.deepcopy(net_x2y.state_dict())
        for epoch in range(epochs):
            adjust_learning_rate(optimizer_x2y, epoch)
            train(net_h2y, net_x2y, optimizer_x2y, loss_func, train_writer_x2y, train_loader, add_physical_info,
                  epoch, stage=2)
            x2y_error = val(net_h2y, net_x2y, loss_func, train_writer_x2y, val_loader, add_physical_info,
                            epoch, stage=2)
            if x2y_error < best_x2y_error:
                best_x2y_error = x2y_error
                best_x2y_model = copy.deepcopy(net_x2y.state_dict())

        torch.save(best_x2y_model, save_x2y_model_path)


def train(model_1, model_2, optimizer, loss_func, train_writer, loader, add_physical_info, epoch, stage):

    model_1.train()
    model_2.train()

    losses = []
    pred, target = [], []
    epoch_start_time = time.time()
    for step, (x, h, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        start_time = time.time()
        if stage == 1:
            prediction = model_1(h)
            loss = loss_func(prediction, y)
        elif stage == 2:
            if add_physical_info:
                model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_stage_1_model))
                model_1.eval()
                _, physical_info = model_1(h)
                model_2.add_physical_info(physical_info)
                prediction = model_2(x)
            else:
                prediction = model_2(x)
            loss = loss_func(prediction, y)
        else:
            print("please input the correct stage")
            return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.item())
        if step == 0:
            pred = prediction.detach().numpy()
            target = y.detach().numpy()
        else:
            pred = np.concatenate((np.array(pred), prediction.detach().numpy()), 0)
            target = np.concatenate((np.array(target), y.detach().numpy()), 0)
        print(
            f"Stage: {stage}\t Epoch: {epoch} \t Batch_num: {step} \t Loss={loss.data.cpu():.4} \t "
            f"Time={time.time() - start_time:.4}")
    error = np.linalg.norm(target - pred, 2) / np.linalg.norm(target, 2)
    print(f"Train \t Epoch={epoch} \t AVG_Loss={np.mean(losses):.4} \t Time={time.time() - epoch_start_time:.4} \t"
          f"l2_error={error:.4}")
    train_writer.add_scalar('Loss/train', np.mean(losses), epoch)
    train_writer.add_scalar('l2_error/train', error, epoch)
    train_writer.flush()
    # return np.mean(losses)


def val(model_1, model_2, loss_func, train_writer, loader, add_physical_info, epoch, stage):
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        losses = []
        pred, target = [], []
        epoch_start_time = time.time()
        for step, (x, h, y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            start_time = time.time()
            if stage == 1:
                prediction, _ = model_1(h)
                loss = loss_func(prediction, y)
            elif stage == 2:
                if add_physical_info:
                    model_1.load_state_dict(torch.load(model_save_folder + "/%s" % best_stage_1_model))

                    _, physical_info = model_1(h)
                    model_2.add_physical_info(physical_info)
                    prediction = model_2(x)
                else:
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
        print(f"Val \t Epoch={epoch} \t AVG_Loss={np.mean(losses):.4} \t Time={time.time() - epoch_start_time:.4} \t"
              f" l2_error={error:.4}")
        train_writer.add_scalar('Loss/val', np.mean(losses), epoch)
        train_writer.add_scalar('l2_error/val', error, epoch)
        train_writer.flush()
    return error


if __name__ == '__main__':
   main()