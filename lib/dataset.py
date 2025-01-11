'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os 
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
import re
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
# from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis, UrbanBISDatasetConfig
from data.urbanbis.model_util_urbanbis import ScannetDatasetConfig, rotate_aligned_boxes, rotate_aligned_boxes_along_axis, center_and_scale
# data setting
# DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
# MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
MEAN_COLOR_RGB = np.array([88.4, 83.9, 77.6])
# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
# MULTIVIEW_DATA = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
# no-multiview
# MULTIVIEW_DATA = CONF.MULTIVIEW

GLOVE_PICKLE = os.path.join("/mnt/d/ScanRefer-master/data", "glove.p")
DC = ScannetDatasetConfig()

#jinxizhi add 2024-10-14
class UrbanReferDataset(Dataset):
    def __init__(self, urbanrefer, urbanrefer_all_scene,
        split="train", 
        num_points=40000, 
        use_color=False, 
        use_normal=False,
        augment=False):

        self.urbanrefer = urbanrefer
        self.urbanrefer_all_scene = urbanrefer_all_scene  # UrbanRefer 数据
        self.split = split
        self.num_points = num_points
        self.use_color = use_color
        self.use_normal = use_normal
        self.augment = augment
        self.scene_id_to_number = json.load(open("/mnt/d/our/data/urbanbis/meta_data/scene_id_to_number.json", "r"))

        # 加载数据
        self._load_data()

    def __len__(self):
        return len(self.urbanrefer)

    def __getitem__(self, idx):
        start = time.time()
        # 根据 UrbanRefer 数据格式加载场景和建筑物信息
        scene_id = self.urbanrefer[idx]["scene_id"]
        building_id = int(self.urbanrefer[idx]["building_id"])
        object_name = " ".join(self.urbanrefer[idx]["building_name"].split("_"))
        # building_ids = [item["building_id"] for item in self.data]
        # semantic_label = [item["Semantic_label"] for item in self.data]
        # instance_bbox = [item["bbox"] for item in self.data]
        '''
        bbox_center = np.array(instance_bbox["center"])  # 边界框中心点
        bbox_size = np.array(instance_bbox["size"])  # 边界框大小
        # new
        instance_bboxes = [bbox_center, bbox_size, [semantic_label], [building_ids]]
        instance_bboxes = np.concatenate(instance_bboxes)  # 将所有子列表拼接成一个数组
        '''

        # 加载语言特征
        lang_feat = self.lang[scene_id][str(building_id)]  # 根据 scene_id, area_id 和 building_id 来加载
        lang_len = len(self.urbanrefer[idx]["description"].split())  # UrbanRefer 里使用 "description"
        lang_len = min(lang_len, CONF.TRAIN.MAX_DES_LEN)
        #仅仅点云
        mesh_vertices = self.scene_data[scene_id]["mesh_vertices"]

        instance_labels = self.scene_data[scene_id]["instance_labels"]

        semantic_labels = self.scene_data[scene_id]["semantic_labels"]

        # fine_grained_categories = self.scene_data[scene_id][area_id]["fine_grained_categories"]
        instance_bboxes = self.scene_data[scene_id]["instance_bboxes"]
        
        # 根据需要加载颜色特征
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # 仅使用 XYZ 坐标
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]  # 使用 XYZ 和 RGB 颜色信息
            point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
            pcl_color = mesh_vertices[:, 3:6]


        # 对点云进行随机采样
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        # fine_grained_categories = fine_grained_categories[choices]
        pcl_color = pcl_color[choices]
        

        # --------------------labels-----------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))

        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        ref_box_label = np.zeros(MAX_NUM_OBJ) # bbox label for reference target

        ref_center_label = np.zeros(3) # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3) # bbox size residual for reference target

        if self.split != "test":
            # 如果 num_bbox 的数量超过 MAX_NUM_OBJ 则截断
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            # target_bboxes[0:num_bbox, :3] = bbox_center  # 处理你的 `UrbanRefer` 或 `UrbanBIS` 格式的边界框
            # target_bboxes[0:num_bbox, 3:6] = bbox_size
            target_bboxes[0:num_bbox, :] = instance_bboxes[0:MAX_NUM_OBJ, 0:6]
            # print(target_bboxes[0:num_bbox])
            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- 数据增强部分 ------------------------------
            # 数据增强部分，对点云和边界框数据进行增强，进行平移和翻转  
            if self.augment and not self.debug:
                if np.random.random() > 0.5:
                    # YZ 平面的翻转
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]                
                    
                if np.random.random() > 0.5:
                    # XZ 平面的翻转
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]                                

                # 沿 X 轴旋转
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 度
                rot_mat = rotx(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # 沿 Y 轴旋转
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 度
                rot_mat = roty(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # 沿 Z 轴旋转
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 度
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

                # 平移
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            # 生成投票点，用于模型学习每个点相对于目标中心的位移
            for i_instance in np.unique(instance_labels):            
                # 找到属于同一实例的所有点
                ind = np.where(instance_labels == i_instance)[0]
                # 找到语义标签
                # 调用方法，获取类别映射字典
                if semantic_labels[ind[0]] in DC.nyu40ids:  # 修改为调用后的结果           
                    x = point_cloud[ind, :3] # 获取属于当前实例的所有点的坐标
                    center = 0.5 * (x.min(0) + x.max(0)) # 计算实例的包围盒中心
                    point_votes[ind, :] = center - x # 每个点的投票向量是中心点到该点的差向量。
                    point_votes_mask[ind] = 1.0 # 标记这些点为有效的投票点

            point_votes = np.tile(point_votes, (1, 3))  # 生成三次相同的投票
            # print("DC.nyu40id2class:", DC.nyu40id2class)
            # 使用 semantic_labels 代替 instance_bboxes 中的类别信息
            # print("instance_bboxes[:num_bbox, -2]:", instance_bboxes[:num_bbox, -2])

            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox, -2]]  # 从语义标签中获取类别信息
            # 将类别索引存储到 size_classes 数组中
            size_classes[0:num_bbox] = class_ind

            # 计算每个实例的大小残差值，target_bboxes[:, 3:6] 是 UrbanBIS 中的物体尺寸 (w, h, d)
            # size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind, :]
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6]
            # construct the reference target label for each bbox
            # 生成参考目标框标签
            ref_box_label = np.zeros(MAX_NUM_OBJ)
            for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                if gt_id == building_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]
        else:
            # 如果是测试集，处理方式与训练集不同，不进行数据增强
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9])  # 生成三次相同的投票
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))

        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass

        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 14


        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        data_dict["lang_feat"] = lang_feat.astype(np.float32) # language feature vectors
        data_dict["lang_len"] = np.array(lang_len).astype(np.int64) # length of each description
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)

        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["ref_box_label"] = ref_box_label.astype(np.int64) # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)

        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["building_id"] = np.array(int(building_id)).astype(np.int64)
        # data_dict['scene_id'] = np.array(int(self.scene_id_to_number[scene_id])).astype(np.int64)
        # data_dict["ann_id"] = np.array(int(ann_id)).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        # data_dict["unique_multiple"] = np.array(self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]).astype(np.int64)
        data_dict["pcl_color"] = pcl_color
        data_dict["load_time"] = time.time() - start

        return data_dict
    
    def _get_raw2label(self):
        # mapping
        # scannet_labels = DC.type2class.keys()
        # scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        # lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        # lines = lines[1:]
        # raw2label = {}
        type2class = {'terrain':0, 'vegetation':1, 'water':2, 'bridge':3, 'vehicle':4, 'boat':5,
            'building':6,'commercial building':7,'residential building':8, 'office building':9, 'cultural building':10, 
            'tranportation building':11, 'municipal building':12, 'temporary building':13, 'others': 14}
        nyu40ids2class = {}
        nyu40ids = np.array([ 6, 7, 8, 9, 10, 11, 12, 13])
        for key, value in type2class.items():
            if value in nyu40ids:
                nyu40ids2class[value] = value
            else:
                nyu40ids2class[value] = 14

        return nyu40ids2class


    def _tranform_des(self):
        #
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)
        # 遍历 scanrefer 列表中的每条数据，提取 scene_id（场景 ID）、object_id（物体 ID）和 ann_id（标注 ID）。
        lang = {}
        for data in self.urbanrefer:
            scene_id = data['scene_id']
            building_id = data["building_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if building_id not in lang[scene_id]:
                lang[scene_id][building_id] = {}

            # tokenize the description
            tokens = re.findall(r"\w+|[^\w\s]", data["description"])
            # tokens 是描述的分词列表，embeddings 是一个全零的数组，形状为 (MAX_DES_LEN, 300)，
            # 每个描述最多包含 MAX_DES_LEN 个单词，每个单词对应 300 维的词嵌入。
            embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
            # tokens = ["sos"] + tokens + ["eos"]
            # embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN + 2, 300))
            # 遍历描述的每个token，如果每个token在GloVe中找到，就将其嵌入到embeddings
            # 如果没有找到token，则使用“unk”嵌入
            for token_id in range(CONF.TRAIN.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["unk"]

            # store
            #存储
            lang[scene_id][building_id] = embeddings

        return lang
    '''
    def _load_data(self):
        print("Loading data...")

        # 加载语言描述
        self.lang = self._tranform_des()

        # 收集所有场景 ID 和区域 ID
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.urbanrefer])))

        # 加载每个场景和区域的数据

        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}
            self.scene_data[scene_id]["mesh_vertices"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_vert.npy')
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_sem_label.npy')
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_ins_label.npy')
            self.scene_data[scene_id]["instance_bboxes"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+'_aligned_bbox.npy')
                

        # self.raw2label：通过 _get_raw2label() 函数生成从原始标签到训练类别的映射。
        self.raw2label = self._get_raw2label()
        self.label2raw = {v: k for k, v in self.raw2label.items()}

    '''

    def _load_data(self):
        print("Loading data...")

        # 加载语言描述
        self.lang = self._tranform_des()

        # 收集所有场景 ID
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.urbanrefer])))

        # 加载每个场景和区域的数据
        self.scene_data = {}
        for scene_id in self.scene_list:
            self.scene_data[scene_id] = {}

            # 加载点云数据
            raw_vertices = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + '_vert.npy')

            
            scaled_vertices, center = center_and_scale(raw_vertices[:, :3], scale_factor=0.01)
            self.scene_data[scene_id]["mesh_vertices"] = np.hstack((scaled_vertices, raw_vertices[:, 3:]))  # 保留颜色信息

            # 加载并缩放 instance_bboxes
            raw_bboxes = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + '_aligned_bbox.npy')

            # 缩放包围盒的中心点
            raw_bboxes[:, 0:3] = (raw_bboxes[:, 0:3] - center) * 0.01

            # 缩放包围盒的尺寸（dx, dy, dz）
            raw_bboxes[:, 3:6] = raw_bboxes[:, 3:6] * 0.01

            self.scene_data[scene_id]["instance_bboxes"] = raw_bboxes

            # 加载其他标签数据
            self.scene_data[scene_id]["semantic_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + '_sem_label.npy')
            self.scene_data[scene_id]["instance_labels"] = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + '_ins_label.npy')

        # self.raw2label：通过 _get_raw2label() 函数生成从原始标签到训练类别的映射。
        self.raw2label = self._get_raw2label()
        self.label2raw = {v: k for k, v in self.raw2label.items()}


    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
