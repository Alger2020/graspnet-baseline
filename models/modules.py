""" Modules for GraspNet baseline model.
    Author: chenxi-wang
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import pytorch_utils as pt_utils
from pointnet2_utils import CylinderQueryAndGroup
from loss_utils import generate_grasp_views, batch_viewpoint_params_to_matrix


class ApproachNet(nn.Module):
    def __init__(self, num_view, seed_feature_dim):
        """ Approach vector estimation from seed point features.

            Input:
                num_view: [int]
                    number of views generated from each each seed point
                seed_feature_dim: [int]
                    number of channels of seed point features
        """
        """ApproachNet 的核心任务是为骨干网络（Backbone）输出的每一个“种子点”完成两件事：

                物体性判断 (Objectness Prediction)：判断这个点是属于一个可抓取物体，还是属于背景（如桌面、墙壁）。
                抓取视角估计 (View Estimation)：从一系列预定义的抓取方向（视角）中，为该点选择一个最适合抓取的接近方向。
                可以把它理解为抓取检测的第一步：“在哪里抓，以及从哪个方向接近？”
        """
        super().__init__()
        self.num_view = num_view   #300, 预定义的候选视角数量
        self.in_dim = seed_feature_dim  # 256, 输入特征维度
        # 定义三个1x1卷积层，作用等同于对每个点独立进行MLP操作
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)#这里的1x1卷积是一种在点云处理中非常高效的技巧。当作用于形状为 (B, C, N) 的数据时，它等价于一个全连接层（MLP）独立地应用于每个点（N）的特征向量（C）上，同时在整个批次（B）中共享权重。
        self.conv2 = nn.Conv1d(self.in_dim, 2+self.num_view, 1)
        self.conv3 = nn.Conv1d(2+self.num_view, 2+self.num_view, 1)#这是最关键的预测层。它将256维的特征映射到一个 2 + 300 维的向量。这 302 维的输出将被拆分用于两个不同的预测任务
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(2+self.num_view)

    def forward(self, seed_xyz, seed_features, end_points):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]
                
                seed_xyz: 种子点坐标 (B, 1024, 3)。
                seed_features: 种子点特征 (B, 256, 1024)。
                end_points: 包含中间结果的字典。
                
            Output:
                end_points: [dict]
        """
        B, num_seed, _ = seed_xyz.size()#B:B num_seed:1024
        features = F.relu(self.bn1(self.conv1(seed_features)), inplace=True) #output:(B,256,1024)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)#output:(B,302,1024)
        features = self.conv3(features) #output:(B,302,1024) 
        objectness_score = features[:, :2, :] # (B, 2, 1024)将 (B, 302, 1024) 的输出张量进行拆分,前2个通道代表每个点是“背景”和“物体”的分数。
        view_score = features[:, 2:2+self.num_view, :].transpose(1,2).contiguous() # (B, 1024, 300)将 (B, 302, 1024) 的输出张量进行拆分，后300个通道代表每个点在300个预定义视角下的抓取质量分数。.transpose(1,2) 是为了将形状变为 (B, num_seed, num_view)，方便后续处理。
        end_points['objectness_score'] = objectness_score #（B，2，1024）,2个通道应该是代表该点可不可以抓取，而不是什么"背景""物体"的分数
        end_points['view_score'] = view_score #(B, 1024, 300)300个通道代表每个点在300个预定义视角下的抓取质量分数。

        # print(view_score.min(), view_score.max(), view_score.mean())
        #top_view_scores: 每个点的最佳视角分数
        #top_view_inds: 每个点的最佳视角对应的索引（0-299之间）。
        top_view_scores, top_view_inds = torch.max(view_score, dim=2) # (B, num_seed)对每个种子点（1024个），在300个视角分数中寻找最大值。
         # 准备通过索引来拾取对应的视角向量,形状: (B, 1024, 1, 3)
        top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()#(B,1024,1,3)
        # 生成预定义的300个视角向量
        template_views = generate_grasp_views(self.num_view).to(features.device) # (num_view, 3)(300,3)
          # 将模板扩展到与批次和种子点数量匹配,形状(B,1024,300,3)
        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1, -1).contiguous() #(B, num_seed, num_view, 3)
        
         # 根据最佳视角索引，从300个模板中选出对应的向量
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2) #(B, num_seed, 3)(B,1024,3)
        
        # 将视角向量转换为旋转矩阵
        vp_xyz_ = vp_xyz.view(-1, 3)
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)## 形状: (B, 1024, 3, 3)
        end_points['grasp_top_view_inds'] = top_view_inds #形状(B,1024)，每个点的最佳视角索引（0-299之间)
        end_points['grasp_top_view_score'] = top_view_scores #(B,1024),每个点的最佳视角分数
        end_points['grasp_top_view_xyz'] = vp_xyz #(B,1024,3),根据最佳视角索引，从300个模板中选出对应的向量
        end_points['grasp_top_view_rot'] = vp_rot #(B,1024,3,3),将视角向量vp_xyz转换为旋转矩阵
        
#ApproachNet 的输出，特别是 grasp_top_view_rot，是连接 GraspNetStage1 和 GraspNetStage2 的桥梁。
# GraspNetStage2 中的 CloudCrop 模块将使用这个旋转矩阵来对齐每个抓取点周围的局部点云，
# 从而在一个标准化的坐标系下进行更精细的抓取参数（如张开宽度、抓取深度）估计。

        return end_points



"""CloudCrop 模块整体概述
CloudCrop 模块是连接抓取检测第一阶段（视角估计）和第二阶段（精细参数估计）的关键桥梁。
它的核心功能是：针对第一阶段输出的每个“种子点”及其最佳抓取接近方向，从原始的、完整的场景点云中“裁剪”出多个圆柱体区域内的局部点云。
然后，它使用一个迷你的 PointNet（即 SharedMLP + max_pool2d）来处理这些局部点云，为每个裁剪区域提取一个紧凑的特征向量。
这些特征向量随后将被送入 OperationNet 和 ToleranceNet，用于预测最终的抓取宽度、角度、分数和容差。
"""
class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    """__init__(...): 构造函数，用于初始化模块的各个组件和参数。
    nsample: 整数，定义了在每个圆柱体区域内要采样的点的数量。在 graspnet.py 中，这个值被设为 64。
    seed_feature_dim: 整数，输入点云的特征维度。在 graspnet.py 中，这个值被设为 3，表示只使用点的 (x, y, z) 坐标作为特征。
    cylinder_radius: 浮点数，定义了裁剪圆柱体的半径，默认为 0.05 米（5厘米）。
    hmin: 浮点数，定义了圆柱体相对于种子点的“底部”高度，默认为 -0.02 米（-2厘米）。
    hmax_list: 浮点数列表，定义了多个圆柱体的“顶部”高度。默认列表 [0.01, 0.02, 0.03, 0.04] 意味着对于每一个种子点，都会创建4个不同深度（高度）的圆柱体进行裁剪。
    """
    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]  #定义了一个列表，用于配置一个共享多层感知机（Shared MLP）。
                                            #这个 MLP 的输入维度是 self.in_dim (即3)，然后经过两个隐藏层（维度分别为64和128），最终输出一个256维的特征向量

        self.groupers = []
        for hmax in hmax_list: #对于每一个 hmax，创建一个 CylinderQueryAndGroup 实例并添加到 self.groupers 列表中。CylinderQueryAndGroup 是一个自定义的 PointNet++ 操作，
            self.groupers.append(CylinderQueryAndGroup(  #它负责执行核心的“圆柱体裁剪”任务。这里会创建4个 CylinderQueryAndGroup 实例，每个对应一个不同的裁剪深度。
                cylinder_radius, hmin, hmax, nsample, use_xyz=True   
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True) #使用之前定义的 mlps 列表来创建一个 SharedMLP 模块。这个模块将对裁剪出的局部点云进行特征提取。bn=True 表示在 MLP 的每一层后都使用批归一化（Batch Normalization）。

    def forward(self, seed_xyz, pointcloud, vp_rot):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors

            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        
        """输入 seed_xyz: 种子点的坐标。
        维度: (B, num_seed, 3)，其中 B 是批次大小，num_seed 是种子点数量（例如1024）。
        输入 pointcloud: 完整的、原始的场景点云。
        维度: (B, N, 3)，其中 N 是场景中的总点数（例如20000）。
        输入 vp_rot: 从第一阶段预测出的、用于对齐局部坐标系的旋转矩阵。
        维度: (B, num_seed, 3, 3)。
        B, num_seed, _, _ = vp_rot.size(): 从输入张量的形状中获取批次大小 B 和种子点数量 num_seed。
        num_depth = len(self.groupers): 获取裁剪的深度数量，这里是 4。
        """
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers)    
        grouped_features = []
        
        """grouped_features = []: 初始化一个空列表，用于收集不同深度下裁剪出的点云特征。
        for grouper in self.groupers:: 遍历之前创建的4个 CylinderQueryAndGroup 实例。
        grouped_features.append(grouper(...)): 调用每个 grouper。grouper 会执行以下操作：
        对于 num_seed 个种子点中的每一个，使用 vp_rot 将其周围的 pointcloud 对齐。
        在对齐后的坐标系中，以 seed_xyz 为中心，裁剪出一个半径为 cylinder_radius、高度从 hmin 到 hmax 的圆柱体区域。
        从该区域内采样 nsample (64) 个点。
        输出维度: 每个 grouper 的输出是一个张量，其维度为 (B, 3, num_seed, nsample)。3 是输入点云的特征维度（xyz）。
        """
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot
            )) # (batch_size, feature_dim, num_seed, nsample)
            
        #调整维度，进行类似poinnet操作
        grouped_features = torch.stack(grouped_features, dim=3) # (batch_size, feature_dim, num_seed, num_depth, nsample)
        grouped_features = grouped_features.view(B, -1, num_seed*num_depth, self.nsample) # (batch_size, feature_dim, num_seed*num_depth, nsample)

        vp_features = self.mlps(
            grouped_features
        ) # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        ) # (batch_size, mlps[-1], num_seed*num_depth, 1)
        
        #输出维度: (B, 256, num_seed, num_depth)，即 (B, 256, 1024, 4)。
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        #这个张量为每个种子点（1024个）在每个裁剪深度下（4个）都生成了一个256维的特征向量，它编码了该局部区域的点云几何信息。
        return vp_features #(b,256,1024,4)  1024个点，每个点生成4个尺度的圆柱全局特征

        
class OperationNet(nn.Module):
    """ Grasp configure estimation.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 3*num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)

        # split prediction
        end_points['grasp_score_pred'] = vp_features[:, 0:self.num_angle]
        end_points['grasp_angle_cls_pred'] = vp_features[:, self.num_angle:2*self.num_angle]
        end_points['grasp_width_pred'] = vp_features[:, 2*self.num_angle:3*self.num_angle]
        return end_points

    
class ToleranceNet(nn.Module):
    """ Grasp tolerance prediction.
    
        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """
    def __init__(self, num_angle, num_depth):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed*num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points