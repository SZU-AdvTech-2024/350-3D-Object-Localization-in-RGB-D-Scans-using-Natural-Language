import os
import sys
import json
import argparse
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
# from data.scannet.model_util_scannet import ScannetDatasetConfig, UrbanBISDatasetConfig 
from data.urbanbis.model_util_urbanbis import ScannetDatasetConfig
from lib.dataset import UrbanReferDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet

# 加载 UrbanRefer 数据集
URBANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.URBANREFER, "urbanRefer_text_train.json")))
URBANREFER_VAL = json.load(open(os.path.join(CONF.PATH.URBANREFER, "urbanRefer_text_val.json")))

DC = ScannetDatasetConfig()

def get_dataloader(args, urbanrefer, all_scene_list, split, config, augment):
    dataset = UrbanReferDataset(
        urbanrefer=urbanrefer[split], 
        urbanrefer_all_scene = all_scene_list,
        split=split, 
        num_points=args.num_points, 
        use_color=args.use_color,
        use_normal=args.use_normal
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args):
    # 初始化模型
    input_channels = int(args.use_normal) * 3 + int(args.use_color) * 3
    model = RefNet(
        num_class = DC.num_class,
        num_heading_bin = DC.num_heading_bin,
        num_size_cluster = DC.num_size_cluster,
        mean_size_arr = DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference
    )

    # 训练模型
    if args.use_pretrained:
        # 加载预训练模型
        print("Loading Pretrained VoteNet...")
        pretrained_model = RefNet(
            num_class = DC.num_class,
            num_headding_bin=DC.num_heading_bin,
            num_size_cluster = DC.num_size_cluster,
            mean_size_arr = DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_bidir=args.use_bidir,
            no_reference=True
        )

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # 替换网络结构
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

        if args.no_detection:
            # 冻结网络层
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            for param in model.vgen.parameters():
                param.requires_grad = False

            for param in model.proposal.parameters():
                param.requires_grad = False
    
    # 使用 CUDA 加速
    model = model.cuda()

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):
    model = get_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    LR_DECAY_STEP = [80, 120, 160] if args.no_reference else None
    LR_DECAY_RATE = 0.1 if args.no_reference else None
    BN_DECAY_STEP = 20 if args.no_reference else None
    BN_DECAY_RATE = 0.5 if args.no_reference else None

    solver = Solver(
        model=model,
        config=DC, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference, 
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "city_area_{}.txt".format(split)))])
    #print(scene_list)

    return scene_list

def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_urbanrefer(scanrefer_train, scanrefer_val, num_scenes):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(URBANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(URBANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        
        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    # print(train_scene_list)
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list



def train(args):
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_urbanrefer(URBANREFER_TRAIN, URBANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # 加载数据
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", DC, True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val", DC, False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_parpms, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_parpms, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="训练标签", default="")
    parser.add_argument("--gpu", type=str, help="使用的 GPU", default="7")
    parser.add_argument("--batch_size", type=int, help="批量大小", default=10)
    parser.add_argument("--epoch", type=int, help="训练周期数", default=100)
    parser.add_argument("--verbose", type=int, help="日志显示频率", default=10)
    parser.add_argument("--val_step", type=int, help="验证步数", default=500)
    parser.add_argument("--lr", type=float, help="学习率", default=1e-3)
    parser.add_argument("--wd", type=float, help="权重衰减", default=1e-5)
    parser.add_argument("--num_points", type=int, default=40000, help="点云点数")
    parser.add_argument("--num_proposals", type=int, default=256, help="提议数量")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    # parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # parser.add_argument("--no_height", action="store_true", help="不使用高度信号")
    parser.add_argument("--no_lang_cls", action="store_true", help="不使用语言分类器")
    parser.add_argument("--no_detection", action="store_true", help="不训练检测模块")
    parser.add_argument("--no_reference", action="store_true", help="不训练引用模块")
    parser.add_argument("--use_color", action="store_true", help="使用 RGB 颜色")
    parser.add_argument("--use_normal", action="store_true", help="使用法线")
    # parser.add_argument("--use_multiview", action="store_true", help="使用多视图图像")
    parser.add_argument("--use_bidir", action="store_true", help="使用双向 GRU")
    parser.add_argument("--use_pretrained", type=str, help="使用预训练模型")
    parser.add_argument("--use_checkpoint", type=str, help="使用检查点", default="")
    args = parser.parse_args()

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)

    
