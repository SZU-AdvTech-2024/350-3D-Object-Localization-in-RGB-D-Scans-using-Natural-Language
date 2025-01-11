import os
import sys
from easydict import EasyDict

CONF = EasyDict()

# path
CONF.PATH = EasyDict()
CONF.PATH.BASE = "/mnt/d/our/" # TODO: change this
CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "urbanbis")
# new add
CONF.PATH.URBANBIS = os.path.join("/mnt/Dataset/")
CONF.PATH.URBANREFER = os.path.join("/mnt/d/our/data/Urbanrefer/")
CONF.PATH.LIB = os.path.join(CONF.PATH.BASE, "lib")
CONF.PATH.MODELS = os.path.join(CONF.PATH.BASE, "models")
CONF.PATH.UTILS = os.path.join(CONF.PATH.BASE, "utils")

# append to syspath
for _, path in CONF.PATH.items():
    sys.path.append(path)

# scannet data
CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
CONF.PATH.SCANNET_META = os.path.join("/mnt/d/our/data/urbanbis", "meta_data")
CONF.PATH.SCANNET_DATA = os.path.join("/mnt/d/our/data/urbanbis", "urbanbis_data")


# data
CONF.SCANNET_DIR =  "/mnt/d/ScanRefer-master/data/scannet/scans/" # TODO change this

CONF.SCANNET_FRAMES_ROOT = "/mnt/d/ScanRefer-master/data/frames_square/" # TODO change this

CONF.PROJECTION = "/mnt/d/multiview_projection_scanrefer" # TODO change this

CONF.ENET_FEATURES_ROOT = "/mnt/d/enet_features" # TODO change this

CONF.ENET_FEATURES_SUBROOT = os.path.join(CONF.ENET_FEATURES_ROOT, "{}") # scene_id
CONF.ENET_FEATURES_PATH = os.path.join(CONF.ENET_FEATURES_SUBROOT, "{}.npy") # frame_id
CONF.SCANNET_FRAMES = os.path.join(CONF.SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode 
CONF.SCENE_NAMES = sorted(os.listdir(CONF.SCANNET_DIR))
CONF.ENET_WEIGHTS = os.path.join(CONF.PATH.BASE, "data/scannetv2_enet.pth")
# CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats.hdf5")
CONF.MULTIVIEW = os.path.join(CONF.PATH.SCANNET_DATA, "enet_feats_maxpool.hdf5")
CONF.NYU40_LABELS = os.path.join(CONF.PATH.SCANNET_META, "nyu40_labels.csv")

# scannet
CONF.SCANNETV2_TRAIN = os.path.join(CONF.PATH.SCANNET_META, "city_area_train.txt")
CONF.SCANNETV2_VAL = os.path.join(CONF.PATH.SCANNET_META, "city_area_val.txt")
CONF.SCANNETV2_TEST = os.path.join(CONF.PATH.SCANNET_META, "city_area_test.txt")
CONF.SCANNETV2_LIST = os.path.join(CONF.PATH.SCANNET_META, "city_area.txt")

# output
CONF.PATH.OUTPUT = os.path.join("/mnt/urban/urbanrefer", "outputs")

# train
CONF.TRAIN = EasyDict()
CONF.TRAIN.MAX_DES_LEN = 1024
CONF.TRAIN.SEED = 42
