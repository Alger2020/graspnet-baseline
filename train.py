""" Training routine for GraspNet baseline model. """

import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
from graspnet import GraspNet, get_loss
from pytorch_utils import BNMomentumScheduler
from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
from label_generation import process_grasp_labels



"""train.py 脚本整体概述
train.py 是整个 GraspNet 模型训练流程的入口和总控制器。它的核心职责包括：
1. 配置解析：通过命令行参数接收数据集路径、相机类型、超参数（如批次大小、学习率）等配置。
2. 数据加载：初始化 GraspNetDataset 和 DataLoader，负责从硬盘读取场景数据和预处理过的抓取标签，并以批次（batch）的形式提供给模型。
3. 模型与优化器初始化：创建 GraspNet 模型实例，设置 Adam 优化器，并配置学习率和批归一化（BN）动量的衰减策略。
4. 断点续训：检查是否存在检查点（checkpoint）文件，如果存在，则加载已保存的模型权重和优化器状态，从上次中断的地方继续训练。
5. 训练与评估循环：执行主训练循环。在每个周期（epoch）中，调用 train_one_epoch 在训练集上训练模型，然后调用 evaluate_one_epoch 在测试集上评估模型性能。
6. 日志与保存：使用 TensorBoard 记录训练和评估过程中的各项损失（loss）和精度（accuracy）指标，并在每个周期结束后保存模型检查点。
"""


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
# ------------------------------------------------------------------------- 全局配置设置

EPOCH_CNT = 0   #全局变量，用于跟踪当前训练周期
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]   #将字符串形式的学习率衰减步骤转换为整数列表
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')] #将字符串形式的学习率衰减率转换为浮点数列表
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))  #
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')  #设置默认检查点路径，没有指定就使用默认路径
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

#确保日志目录存在，不存在则创建
if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

#打开日志文件，以追加模式('a')写入
LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
#首先记录配置参数到日志
LOG_FOUT.write("配置参数:"+str(cfgs)+'\n')
#定义log_string函数，同时将信息输出到日志文件和控制台
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders 
#  数据集和数据加载器初始化

#为数据加载器中的每个工作进程设置不同的随机种子，确保数据增强的随机性
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader

#####内存占用相关代码#####
import psutil
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024*1024)
print(f'加载标签前内存占用: {mem_before:.2f} GB')
#####内存占用相关代码#####

valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)  #加载有效对象索引和抓取标签数据

#####内存占用相关代码#####
mem_after = process.memory_info().rss / (1024 * 1024*1024)
print(f'加载标签后内存占用: {mem_after:.2f} GB')
print(f'grasp_labels 大约占用了: {mem_after - mem_before:.2f} GB')
#####内存占用相关代码#####

#创建训练和测试数据集
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, remove_outlier=True, augment=True)

#####内存占用相关代码#####
traindataset_after = process.memory_info().rss / (1024 * 1024*1024)
print(f'创建训练数据集后标签后内存占用: {traindataset_after:.2f} GB')
print(f'创建训练数据集 大约占用了: {traindataset_after - mem_after:.2f} GB')
#####内存占用相关代码#####

TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False)

#####内存占用相关代码#####
testdataset_after = process.memory_info().rss / (1024 * 1024*1024)
print(f'创建测试数据集后标签后内存占用: {testdataset_after:.2f} GB')
print(f'创建测试数据集 大约占用了: {testdataset_after - traindataset_after:.2f} GB')
#####内存占用相关代码#####

print("创建训练集大小:",len(TRAIN_DATASET),"创建测试集大小:",len(TEST_DATASET))

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

#####内存占用相关代码#####
dataloader_after = process.memory_info().rss / (1024 * 1024*1024)
print(f'加载数据集后标签后内存占用: {dataloader_after:.2f} GB')
print(f'加载数据集 大约占用了: {dataloader_after - testdataset_after:.2f} GB')
#####内存占用相关代码#####

print("加载数据集大小：",len(TRAIN_DATALOADER),"创建数据集大小:",len(TEST_DATALOADER))


# Init the model and optimzier
# 模型和优化器初始化

#初始化GraspNet模型，配置参数：
    #输入特征维度：0（仅使用xyz坐标）
    #视角数量：由命令行参数指定 300
    #角度类别数：12
    #深度类别数：4
    #圆柱半径和高度参数
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #选择设备，移至设备
net.to(device)

# Load the Adam optimizer
# 使用Adam优化器，配置初始学习率和权重衰减
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Load checkpoint if there is any
# 检查点加载（断点续训）
it = -1 # 用于 `LambdaLR` 和 `BNMomentumScheduler` 的初始化值
start_epoch = 0  #初始化迭代计数器和起始周期
#如果存在检查点文件，如果有则:
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH) #加载检查点
    net.load_state_dict(checkpoint['model_state_dict']) #恢复模型参数
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #恢复优化器状态
    start_epoch = checkpoint['epoch'] #设置起始周期为检查点中保存的周期
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch)) #记录加载信息

#.批归一化动量调度器设置
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
#设置批归一化动量的初始值和最大值
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
#定义一个lambda函数计算当前迭代的BN动量
#创建BN动量调度器，随着训练进行，动量会从初始值0.5衰减到接近0.999
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

#根据当前周期和预设的衰减步骤计算学习率
def get_current_lr(epoch):
    lr = cfgs.learning_rate
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr
#更新优化器中所有参数组的学习率
def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

#训练一个周期的函数
def train_one_epoch():
    stat_dict = {} # collect statistics 创建统计字典，用于收集训练指标
    adjust_learning_rate(optimizer, EPOCH_CNT) #根据当前周期调整学习率
    bnm_scheduler.step() # decay BN momentum 更新BN动量
    # set model to training mode
    net.train()# 将模型设置为训练模式
    #遍历训练数据加载器的每个批次
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass执行前向传播，得到模型输出的端点字典
        end_points = net(batch_data_label)

        # Compute loss and gradients, update parameters. 
        # 计算损失函数，get_loss函数也会更新端点字典
        loss, end_points = get_loss(end_points)
        loss.backward() #执行反向传播计算梯度
        if (batch_idx+1) % 1 == 0:  #每1个批次更新一次模型参数
            optimizer.step()
            optimizer.zero_grad() #清零梯度，准备下一次迭代

        # Accumulate statistics and print out 
        # 收集和累加各类统计信息（损失、准确率、精度、召回率等）
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10 #每10个批次输出一次日志并记录到TensorBoard
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size)
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval)) #每10个批次输出一次日志并记录到TensorBoard
                stat_dict[key] = 0  #输出平均统计值后重置统计字典

#评估一个周期的函数
def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)

        # Compute loss
        loss, end_points = get_loss(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

    for key in sorted(stat_dict.keys()):
        TEST_WRITER.add_scalar(key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    mean_loss = stat_dict['loss/overall_loss']/float(batch_idx+1)
    return mean_loss

# 主训练函数
def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, cfgs.max_epoch):  #主训练循环，从start_epoch开始到max_epoch结束
        EPOCH_CNT = epoch  #更新全局周期计数器
        #记录当前周期、学习率、BN动量和时间戳
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed() #重置NumPy随机种子以确保数据增强的随机性
        train_one_epoch() #执行一个训练周期
        loss = evaluate_one_epoch()  #执行一个评估周期并获取损失
        # Save checkpoint
        #保存检查点，包含：
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1 下一周期的索引
                    'optimizer_state_dict': optimizer.state_dict(), #优化器状态
                    'loss': loss, #当前损失
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel 多gpu 处理模型是否使用DataParallel包装的情况
            save_dict['model_state_dict'] = net.module.state_dict() #模型参数状态
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar')) #将检查点保存到指定路径

if __name__=='__main__':
    train(start_epoch)
