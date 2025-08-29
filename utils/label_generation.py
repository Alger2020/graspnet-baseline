""" Dynamically generate grasp labels during training.
    Author: chenxi-wang
"""

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'knn'))

from knn_modules import knn
from loss_utils import GRASP_MAX_WIDTH, batch_viewpoint_params_to_matrix,\
                       transform_point_cloud, generate_grasp_views



"""文件核心目标
标签对齐：将存储在每个物体局部坐标系下的抓取标签（抓取点、姿态、分数等），转换到相机/场景的全局坐标系下。
标签分配：为骨干网络（Backbone）输出的每一个种子点（seed point），从场景中所有物体的所有抓取标签里，找到一个最接近、最匹配的标签作为其“真实标签”（Ground Truth）。
标签处理：对分配好的标签进行数值处理（如对数变换），使其更适合作为损失函数的目标。
视角切片：根据 GraspNetStage1 预测的最佳视角，从完整的抓取标签中提取出对应的数据，用于 GraspNetStage2 的训练。

"""

# 为fp2_xyz（骨干网络输出的1024个种子点）中的每一个点，分配一个最完整的的抓取标签(ground truth)。
def process_grasp_labels(end_points):
    """ Process labels according to scene points and object poses. """
    clouds = end_points['input_xyz'] #(B, N, 3)  (B,20000,3) 输入的原始点云
    seed_xyzs = end_points['fp2_xyz'] #(B, Ns, 3) pointnet处理后的(B,1024,3)
    batch_size, num_samples, _ = seed_xyzs.size()

# 初始化用于存储整个批次结果的列表
    batch_grasp_points = []
    batch_grasp_views = []
    batch_grasp_views_rot = []
    batch_grasp_labels = []
    batch_grasp_offsets = []
    batch_grasp_tolerance = []
    
     # 外层循环：遍历批次中的每个场景
    for i in range(len(clouds)): #遍历B
        seed_xyz = seed_xyzs[i] #(Ns, 3) (1024,3)
        poses = end_points['object_poses_list'][i] #(9 , 3 , 4) #poses (每张图片(场景)中每个物体的位姿，比如一张图片中9个物体)

        #  
        grasp_points_merged = []
        grasp_views_merged = []
        grasp_views_rot_merged = []
        grasp_labels_merged = []
        grasp_offsets_merged = []
        grasp_tolerance_merged = []
        
         #  内层循环：遍历当前场景中的每个物体，将其标签转换到场景坐标系
        for obj_idx, pose in enumerate(poses):
            grasp_points = end_points['grasp_points_list'][i][obj_idx] #(Np, 3) (252,3)  这是真实抓取点，np有变化，不是pointnet处理的点
            grasp_labels = end_points['grasp_labels_list'][i][obj_idx] #(Np, V, A, D)  (252,300,12,4) 12是预设角度，4是深度
            grasp_offsets = end_points['grasp_offsets_list'][i][obj_idx] #(Np, V, A, D, 3)
            grasp_tolerance = end_points['grasp_tolerance_list'][i][obj_idx] #(Np, V, A, D)
            _, V, A, D = grasp_labels.size()
            num_grasp_points = grasp_points.size(0)
            # generate and transform template grasp views
            grasp_views = generate_grasp_views(V).to(pose.device) #(V, 3)
            grasp_points_trans = transform_point_cloud(grasp_points, pose, '3x4')
            grasp_views_trans = transform_point_cloud(grasp_views, pose[:3,:3], '3x3')
            # generate and transform template grasp view rotation
            angles = torch.zeros(grasp_views.size(0), dtype=grasp_views.dtype, device=grasp_views.device)
            grasp_views_rot = batch_viewpoint_params_to_matrix(-grasp_views, angles) #(V, 3, 3)
            grasp_views_rot_trans = torch.matmul(pose[:3,:3], grasp_views_rot) #(V, 3, 3)
            
            # assign views
            grasp_views_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)
            grasp_views_trans_ = grasp_views_trans.transpose(0, 1).contiguous().unsqueeze(0)
            view_inds = knn(grasp_views_trans_, grasp_views_, k=1).squeeze() - 1
            grasp_views_trans = torch.index_select(grasp_views_trans, 0, view_inds) #(V, 3)
            grasp_views_trans = grasp_views_trans.unsqueeze(0).expand(num_grasp_points, -1, -1) #(Np, V, 3)
            grasp_views_rot_trans = torch.index_select(grasp_views_rot_trans, 0, view_inds) #(V, 3, 3)
            grasp_views_rot_trans = grasp_views_rot_trans.unsqueeze(0).expand(num_grasp_points, -1, -1, -1) #(Np, V, 3, 3)
            grasp_labels = torch.index_select(grasp_labels, 1, view_inds) #(Np, V, A, D)
            grasp_offsets = torch.index_select(grasp_offsets, 1, view_inds) #(Np, V, A, D, 3)
            grasp_tolerance = torch.index_select(grasp_tolerance, 1, view_inds) #(Np, V, A, D)
            # add to list
            grasp_points_merged.append(grasp_points_trans)
            grasp_views_merged.append(grasp_views_trans)
            grasp_views_rot_merged.append(grasp_views_rot_trans)
            grasp_labels_merged.append(grasp_labels)
            grasp_offsets_merged.append(grasp_offsets)
            grasp_tolerance_merged.append(grasp_tolerance)


        grasp_points_merged = torch.cat(grasp_points_merged, dim=0) #(Np', 3)
        grasp_views_merged = torch.cat(grasp_views_merged, dim=0) #(Np', V, 3)
        grasp_views_rot_merged = torch.cat(grasp_views_rot_merged, dim=0) #(Np', V, 3, 3)
        grasp_labels_merged = torch.cat(grasp_labels_merged, dim=0) #(Np', V, A, D)
        grasp_offsets_merged = torch.cat(grasp_offsets_merged, dim=0) #(Np', V, A, D, 3)
        grasp_tolerance_merged = torch.cat(grasp_tolerance_merged, dim=0) #(Np', V, A, D)

        # 为每个种子点分配标签 (核心步骤)
        # 使用KNN找到每个种子点(seed_xyz)在所有真实抓取点(grasp_points_merged)中最近的那个
        # compute nearest neighbors
        seed_xyz_ = seed_xyz.transpose(0, 1).contiguous().unsqueeze(0) #(1, 3, Ns)
        grasp_points_merged_ = grasp_points_merged.transpose(0, 1).contiguous().unsqueeze(0) #(1, 3, Np')
        nn_inds = knn(grasp_points_merged_, seed_xyz_, k=1).squeeze() - 1 #(Ns)


        # assign anchor points to real points
        grasp_points_merged = torch.index_select(grasp_points_merged, 0, nn_inds) # (Ns, 3)
        grasp_views_merged = torch.index_select(grasp_views_merged, 0, nn_inds) # (Ns, V, 3)
        grasp_views_rot_merged = torch.index_select(grasp_views_rot_merged, 0, nn_inds) #(Ns, V, 3, 3)
        grasp_labels_merged = torch.index_select(grasp_labels_merged, 0, nn_inds) # (Ns, V, A, D)
        grasp_offsets_merged = torch.index_select(grasp_offsets_merged, 0, nn_inds) # (Ns, V, A, D, 3)
        grasp_tolerance_merged = torch.index_select(grasp_tolerance_merged, 0, nn_inds) # (Ns, V, A, D)

        # add to batch
        # 将处理好的单一样本结果添加到批次列表中
        batch_grasp_points.append(grasp_points_merged)  
        batch_grasp_views.append(grasp_views_merged) 
        batch_grasp_views_rot.append(grasp_views_rot_merged)
        batch_grasp_labels.append(grasp_labels_merged)
        batch_grasp_offsets.append(grasp_offsets_merged)
        batch_grasp_tolerance.append(grasp_tolerance_merged)


    #至此，我们成功地为1024个种子点中的每一个，都赋予了一个完整的抓取标签。
    batch_grasp_points = torch.stack(batch_grasp_points, 0) #(B, Ns, 3·)  (B,1024,3)  1024个种子点云对应的最接近的真实点云坐标
    batch_grasp_views = torch.stack(batch_grasp_views, 0) #(B, Ns, V, 3)  (B,1024,300,3)  1024个种子点云对应的最真实点云坐标的视角接近向量
    batch_grasp_views_rot = torch.stack(batch_grasp_views_rot, 0) #(B, Ns, V, 3, 3) (B,1024,300,3,3) 对应的接近向量转换的旋转矩阵
    batch_grasp_labels = torch.stack(batch_grasp_labels, 0) #(B, Ns, V, A, D) (B,1024,300,12,4)  
    batch_grasp_offsets = torch.stack(batch_grasp_offsets, 0) #(B, Ns, V, A, D, 3) (B,1024,300,12,4,3) ,这个3有宽度的内容
    batch_grasp_tolerance = torch.stack(batch_grasp_tolerance, 0) #(B, Ns, V, A, D) 

    # process labels     对标签进行数值处理，使其适合作为损失函数的目标
    batch_grasp_widths = batch_grasp_offsets[:,:,:,:,:,2]   
    label_mask = (batch_grasp_labels > 0) & (batch_grasp_widths <= GRASP_MAX_WIDTH)
    u_max = batch_grasp_labels.max()
    batch_grasp_labels[label_mask] = torch.log(u_max / batch_grasp_labels[label_mask])
    batch_grasp_labels[~label_mask] = 0
    # 计算每个视角的最佳分数，用于监督Stage1的视角预测
    batch_grasp_view_scores, _ = batch_grasp_labels.view(batch_size, num_samples, V, A*D).max(dim=-1)


    #为1024个种子点中的每一个，都赋予了一个完整的抓取标签。
    end_points['batch_grasp_point'] = batch_grasp_points      #(B, Ns, 3·)  (B,1024,3)  1024个种子点云对应的最接近的真实点云坐标
    end_points['batch_grasp_view'] = batch_grasp_views     #(B, Ns, V, 3)  (B,1024,300,3)  1024个种子点云对应的最真实点云坐标的视角接近向量
    end_points['batch_grasp_view_rot'] = batch_grasp_views_rot     #(B, Ns, V, 3, 3) (B,1024,300,3,3) 对应的接近向量转换的旋转矩阵
    end_points['batch_grasp_label'] = batch_grasp_labels     #(B, Ns, V, A, D) (B,1024,300,12,4)  
    end_points['batch_grasp_offset'] = batch_grasp_offsets     #(B, Ns, V, A, D, 3) (B,1024,300,12,4,3) ,这个3有宽度的内容
    end_points['batch_grasp_tolerance'] = batch_grasp_tolerance    #(B, Ns, V, A, D)  
    end_points['batch_grasp_view_label'] = batch_grasp_view_scores.float()         # (B,Ns,V) (B,1024,300)计算每个视角的最佳分数，用于监督Stage1的视角预测

    return end_points


# 在 GraspNetStage2 的训练过程中调用。
# 它接收 GraspNetStage1 预测出的最佳视角索引，并用这些索引从 process_grasp_labels 生成的密集标签中，切片出对应的标签。
def match_grasp_view_and_label(end_points):
    """ Slice grasp labels according to predicted views. """
    top_view_inds = end_points['grasp_top_view_inds'] # (B, Ns)
    template_views_rot = end_points['batch_grasp_view_rot'] # (B, Ns, V, 3, 3)
    grasp_labels = end_points['batch_grasp_label'] # (B, Ns, V, A, D)
    grasp_offsets = end_points['batch_grasp_offset'] # (B, Ns, V, A, D, 3)
    grasp_tolerance = end_points['batch_grasp_tolerance'] # (B, Ns, V, A, D)

    B, Ns, V, A, D = grasp_labels.size()
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, 3, 3)
    top_template_views_rot = torch.gather(template_views_rot, 2, top_view_inds_).squeeze(2)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1).expand(-1, -1, -1, A, D)
    top_view_grasp_labels = torch.gather(grasp_labels, 2, top_view_inds_).squeeze(2)
    top_view_grasp_tolerance = torch.gather(grasp_tolerance, 2, top_view_inds_).squeeze(2)
    top_view_inds_ = top_view_inds.view(B, Ns, 1, 1, 1, 1).expand(-1, -1, -1, A, D, 3)
    top_view_grasp_offsets = torch.gather(grasp_offsets, 2, top_view_inds_).squeeze(2)

#返回切片后的标签，同时更新 end_points 字典。视角已经选定，所以去掉视角维度V
    end_points['batch_grasp_view_rot'] = top_template_views_rot  #(B,Ns,3,3)
    end_points['batch_grasp_label'] = top_view_grasp_labels     #(B,Ns,A,D)
    end_points['batch_grasp_offset'] = top_view_grasp_offsets   #(B,Ns,A,D,3)
    end_points['batch_grasp_tolerance'] = top_view_grasp_tolerance #(B,Ns,A,D)

    return top_template_views_rot, top_view_grasp_labels, top_view_grasp_offsets, top_view_grasp_tolerance, end_points