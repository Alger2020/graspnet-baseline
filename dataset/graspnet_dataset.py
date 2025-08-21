""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
# from torch._six import container_abcs

import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points

class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True):
        assert(num_points<=50000)
        self.root = root
        self.split = split                                    # 数据集分割：'train', 'test', 'test_seen'等
        self.num_points = num_points                          # 采样的点云数量 (≤20000)
        self.remove_outlier = remove_outlier                  # 是否移除异常点
        self.remove_invisible = remove_invisible              # 是否移除不可见点
        self.valid_obj_idxs = valid_obj_idxs                  # 有效物体索引
        self.grasp_labels = grasp_labels                      # 抓取标签
        self.camera = camera                                  # 相机类型('kinect'等)
        self.augment = augment                                # 是否数据增强
        self.load_label = load_label                          # 是否加载标签
        self.collision_labels = {}

# 训练集: scene_0000 ~ scene_0099 (100个场景)
# 测试集: scene_0100 ~ scene_0189 (90个场景)
# test_seen: scene_0100 ~ scene_0129
# test_similar: scene_0130 ~ scene_0159
# test_novel: scene_0160 ~ scene_0189
# RGB图像 (rgb/xxxx.png)     (720,1280,3)
# 深度图像 (depth/xxxx.png)   (720,1280)  每个位置就是深度距离数值
# 分割标签 (label/xxxx.png)    (720,1280)  每个位置为所属物体id  object_id_list中的值
# 元数据 (meta/xxxx.mat)   
##      mat中有对应每一帧图片中物体的类别索引cls_indexes、物体的位置姿态poses、相机内参intrinsic_matrix、深度图像中的像素值进行标准化或缩放的因子factor_depth。
##      poses (3,4,9)3表示每个物体的姿态信息在三维空间中的三个分量；4表示每个姿态的四元数表示法中的四个参数，用于描述物体的旋转；9意味当前帧中对应着场景中的 9 个物体
##      cls_indexes (1,9)  该图片场景中物体的9个类别索引
##      intrinsic_matrix (3,3)  相机内参矩阵
##      factor_depth (1,1)     缩放因子，深度图像中的像素值通常是相机到场景中物体的距离，但这些距离通常以某种方式进行了标准化或缩放


# 主要方法
 ##  get_data_label(index) - 获取带标签的数据：
 ##  图像加载: 读取RGB、深度、分割图像和元数据
 ##  点云生成: 从深度图像生成3D点云
 ##  点云采样: 随机采样到指定点数
 ##  标签处理: 加载抓取点、偏移、分数和容差标签
 ##  碰撞检测: 处理碰撞标签，将有碰撞的抓取点分数设为0
 ##  可见性过滤: 移除不可见的抓取点
     
 ##  augment_data() - 数据增强：
 ##  翻转: 沿YZ平面翻转
 ##  旋转: 绕Z轴旋转±30度主要方法

        if split == 'train':
            self.sceneIds = list( range(100) )
        elif split == 'test':
            self.sceneIds = list( range(100,190) )
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,160) )
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190) )
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)

        return ret_dict

# 主要的方法  get_data_label(index) -获取带标签的数据

#   1.图像加载: 读取RGB、深度、分割图像和元数据
#   2.点云生成: 从深度图像生成3D点云
#   3.点云采样: 随机采样到指定点数
#   4.标签处理: 加载抓取点、偏移、分数和容差标签
#   5.碰撞检测: 处理碰撞标签，将有碰撞的抓取点分数设为0
#   6.可见性过滤: 移除不可见的抓取点

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0   #加载RGB图像，转换为浮点数组并归一化到[0,1]范围
        depth = np.array(Image.open(self.depthpath[index]))                             #加载深度图像，保持深度图片原始深度值
        seg = np.array(Image.open(self.labelpath[index]))                               #加载分割标签图像，每个像素值代表物体ID
        meta = scio.loadmat(self.metapath[index])                                       #加载元数据文件(.mat格式)，包含相机内参、物体姿态等信息
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)                   #场景中物体的类别索引 [9]  9个物体的索引
            poses = meta['poses']                                                       #物体的6D姿态  (3,4,9) 3,4表示物体姿态，9表示物体类数
            intrinsic = meta['intrinsic_matrix']                                        #相机内参矩阵  (3,3)
            factor_depth = meta['factor_depth']                                         #深度值缩放因子  （1，1）[1000]
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth) #创建相机信息对象，包含分辨率、焦距、主点坐标和深度因子


        # 生成点云
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)     # （720，1280，3）从深度图像生成3D点云，使用相机内参进行投影变换 
        # print(f"Cloud shape: {cloud.shape}")  # 打印 cloud 的形状


        # 有效点筛选   
        depth_mask = (depth > 0)         #创建布尔掩码，大于零为有效值
        seg_mask = (seg > 0)              #这个逻辑上有物体哎，分割标签的每个像素值代码物体ID，你这样不是把物体ID等于0的也掩掉了吗。还好没用上
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))  
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)         # 如果启用异常点移除：
        else:                                            #  加载相机姿态和对齐矩阵，计算当前帧的变换矩阵，生成工作空间掩码，过滤桌面外的点，组合深度掩码和工作空间掩码
            mask = depth_mask                            #  否则只使用深度掩码
        cloud_masked = cloud[mask]                       #应用掩码，只保留有效的点云、颜色和分割标签   
        color_masked = color[mask]                       # cloud_masked shape:(379278,3)  ,筛掉很多了原来720*1280=921600-->379278
        seg_masked = seg[mask]                           # color_masked shape:(379278,3)                      
                                                         # seg_masked shape:(379278,)
        # 打印筛选后的点云、颜色、分割数据的形状
        # print(f"cloud_masked shape: {cloud_masked.shape}")
        # print(f"color_masked shape: {color_masked.shape}")
        # print(f"seg_masked shape: {seg_masked.shape}")


        # 点云采样，按要求的数量num_points采样
        if len(cloud_masked) >= self.num_points:        # 如果点数足够：随机无重复采样到目标点数 
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:                                           # 如果点数不够：先取全部点，再有重复地随机采样补足
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]              # 根据采样索引获取最终的点云、颜色和分割数据
        color_sampled = color_masked[idxs]              # cloud_sampled shape: (num_points(20000), 3)
        seg_sampled = seg_masked[idxs]                  # color_sampled shape: (num_points(20000), 3)
        objectness_label = seg_sampled.copy()           # seg_sampled shape:(num_poinst(20000))
        objectness_label[objectness_label>1] = 1
        # 打印最终的点云、颜色、分割数据的形状
        # print(f"cloud_sampled shape: {cloud_sampled.shape}")
        # print(f"color_sampled shape: {color_sampled.shape}")
        # print(f"seg_sampled shape: {seg_sampled.shape}")
       
        
        object_poses_list = []
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_scores_list = []
        grasp_tolerance_list = []
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue
            if (seg_sampled == obj_idx).sum() < 50:
                continue
            object_poses_list.append(poses[:, :, i])
            points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
            collision = self.collision_labels[scene][i] #(Np, V, A, D)


            # remove invisible grasp points
            if self.remove_invisible:
                visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
                points = points[visible_mask]
                offsets = offsets[visible_mask]
                scores = scores[visible_mask]
                tolerance = tolerance[visible_mask]
                collision = collision[visible_mask]

            idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
            grasp_points_list.append(points[idxs])
            grasp_offsets_list.append(offsets[idxs])
            collision = collision[idxs].copy()
            scores = scores[idxs].copy()
            scores[collision] = 0
            grasp_scores_list.append(scores)
            tolerance = tolerance[idxs].copy()
            tolerance[collision] = 0
            grasp_tolerance_list.append(tolerance)
        
        if self.augment:
            cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)
        
        ret_dict = {}                                                          # 数据输出格式     
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)            # 3D点云(N,3)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)
        ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        ret_dict['object_poses_list'] = object_poses_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list

        return ret_dict

def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
        if i == 18: continue
        valid_obj_idxs.append(i + 1) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        tolerance = np.load(os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32), tolerance)

    return valid_obj_idxs, grasp_labels

def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

if __name__ == "__main__":
    root = './graspnet'
    valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetDataset(root, valid_obj_idxs, grasp_labels, split='train', remove_outlier=True, remove_invisible=True, num_points=20000)
    print(len(train_dataset))

    end_points = train_dataset[233]
    cloud = end_points['point_clouds']
    seg = end_points['objectness_label']
    print(cloud.shape)
    print(cloud.dtype)
    print(cloud[:,0].min(), cloud[:,0].max())
    print(cloud[:,1].min(), cloud[:,1].max())
    print(cloud[:,2].min(), cloud[:,2].max())
    print(seg.shape)
    print((seg>0).sum())
    print(seg.dtype)
    print(np.unique(seg))
