""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval  #这是官方提供的评估工具包，GraspGroup 用于方便地操作一组抓取姿态，GraspNetEval 是执行评估的核心类。

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset, collate_fn
from collision_detector import ModelFreeCollisionDetector #一个基于体素的、无需模型信息的碰撞检测器。


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root')
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--dump_dir', required=True, help='Dump dir to save outputs')
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=30, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()




"""test.py 是用于评估已训练好的 GraspNet 模型性能的脚本。它包含两个主要功能：

推理 (Inference)：加载一个训练好的模型检查点（checkpoint），遍历测试数据集中的所有场景，对每个场景生成抓取姿态预测。
这些预测会经过碰撞检测过滤，然后保存到指定的输出目录中。

评估 (Evaluation)：使用官方的 graspnetAPI 工具，读取第一步中保存的所有预测结果，并与数据集中的真实标签进行比较，
计算出标准的抓取检测评估指标，主要是平均精度（Average Precision, AP）。 

"""
# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
# ------------------------------------------------------------------------- 全局配置

if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)   #如果指定的输出目录不存在，则创建它。

# Init datasets and dataloaders 
#为数据加载器中的每个工作进程设置不同的随机种子，确保数据增强的随机性
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
#创建测试数据集实例。注意 split='test' 表示使用整个测试集，load_label=False 表示在推理时不需要加载真值标签，节省内存和时间
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test', camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False)

print("创建测试集大小:",len(TEST_DATASET))

#获取测试集中所有场景的名称列表，用于后续保存结果时构建正确的目录结构。
SCENE_LIST = TEST_DATASET.scene_list()
#创建数据加载器，shuffle=False 确保按顺序处理场景。
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

print("加载测试集大小：",len(TEST_DATALOADER))

# Init the model
# 初始化模型
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                     cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Load checkpoint
#加载指定的 .tar 检查点文件
checkpoint = torch.load(cfgs.checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
print("-> 加载检查点模型成功： %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

# 函数 (推理)
def inference():
    batch_interval = 100
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval() #将模型设置为评估模式。这会禁用 Dropout 等层，并固定批归一化层的行为。
    tic = time.time() #开始计时。
    for batch_idx, batch_data in enumerate(TEST_DATALOADER): #遍历测试数据集中的每一个批次（通常每个批次只包含一个场景）
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device) #将输入数据（主要是点云）移动到 GPU。
        
        # Forward pass
        with torch.no_grad(): #在这个上下文管理器中，PyTorch 不会计算梯度，可以显著加快推理速度并减少显存占用
            end_points = net(batch_data)
            #调用解码函数，将网络的原始输出（分数、角度、宽度等预测）转换成一个抓取姿态列表
            grasp_preds = pred_decode(end_points) #是一个列表，每个元素对应批次中一个样本的所有预测抓取，形状为 (num_grasps, 17)。
            
        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds) #使用 graspnetAPI 中的 GraspGroup 类来封装 NumPy 形式的抓取预测，方便后续操作。

            # collision detection
            # 如果设置了碰撞阈值，则执行碰撞检测。
            if cfgs.collision_thresh > 0:
                cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True) #从数据集中获取当前场景的原始、完整的点云。
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size) #使用该点云初始化一个无模型的碰撞检测器。
                #检测 GraspGroup 中的每一个抓取是否会与场景点云发生碰撞。返回一个布尔掩码。
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                #使用掩码过滤掉发生碰撞的抓取。
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(cfgs.dump_dir, SCENE_LIST[data_idx], cfgs.camera) #构建保存路径
            save_path = os.path.join(save_dir, str(data_idx%256).zfill(4)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0: #每隔 100 个批次，打印一次处理时间和当前进度。
            toc = time.time()
            print('Eval batch: %d, time: %fs'%(batch_idx, (toc-tic)/batch_interval))
            tic = time.time()

#函数 (评估)
def evaluate():
    ge = GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split='test') #初始化API提供官方评估器，指定数据集根目录、相机和评估集（test）。
    
    # 调用核心评估方法。它会自动查找 cfgs.dump_dir 目录下的所有预测结果（.npy 文件），加载对应的真值标签，然后使用 cfgs.num_workers 个进程并行计算 AP 分数。
    res, ap = ge.eval_all(cfgs.dump_dir, proc=cfgs.num_workers)
    save_dir = os.path.join(cfgs.dump_dir, 'ap_{}.npy'.format(cfgs.camera))
    np.save(save_dir, res) #将计算出的详细 AP 结果（res）保存到输出目录下的 ap_camera.npy 文件中。

if __name__=='__main__':
    inference()
    evaluate()
