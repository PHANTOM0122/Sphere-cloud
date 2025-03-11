import poselib
import pandas as pd
import os

# Set computing resource
RESOURCE = 'local' # 'alpha' or 'local'

## PATH to DATASET
DATASET_MOUNT = 'data' ## Path to dataset

# GPU, CPU usage options
CUDA = 'cuda:0' ## ['cpu or 'cuda:0']

# SPARSITY LEVEL should have at least one value for any process to run
SPARSITY_LEVEL = [1.0]
### SPARSITY_LEVEL = [1.0, 0.5, 0.25, 0.1]

# NOISE LEVEL should have at least one value for any process to run
NOISE_LEVEL = [0.0]

##################################################
############## Line type selection ###############
##################################################
LINE_TYPE = ["Spherecloud"]

##################################################
########## Options for Sphere cloud ###########
##################################################

## | PIPELINE_TYPE | PGT_TYPE |
## |-----------------------------------------------------------------------------------
## | Relocalization: 'old_gt_triangulated' -> 7Scenes  'old_gt_retriangulated' -> 12Scenes | From Brachmann et al. (ICCV 2021)
## | Recovery: 'pgt_triangulated' -> 7Scenes, 12Scenes 

PIPELINE_TYPE = 'recovery' ## ['relocalization', 'recovery']
PGT_TYPE = 'pgt_triangulated'
USE_CENTROID = True

## Ratio of True Positive (TP) in sphere cloud
TP_RATIO = 25 ## Select from 100%, 50%, 33%, 25%
LAMBDA1 = 1.0
LAMBDA2 = 1e-4

# Raw depth images were used
DEPTH = 'RAW' 

# Depth oracle case
USE_DEPTH_ORACLE = False

##################################################
########### Camera relocalization error ##########
##################################################
# Turn on off the refinement on pose estimation
REFINE_OPTION = True
RANDOM_SEED = 0

# Option for minimal solver for pose estimation 
POSE_SOLVER = 'vp3p' ## 'p6l', 'vp3p

# Poselib options
BUNDLE_OPTIONS = poselib.BundleOptions()
RANSAC_OPTIONS = poselib.RansacOptions()
EPIPOLAR_THRE = 1.5
RANSAC_OPTIONS["max_epipolar_error"] = EPIPOLAR_THRE
RANSAC_OPTIONS["min_iterations"] = 10

## For Enhanced sphere cloud (50%, 33%, 25% TP)
if TP_RATIO != 100:
    RANSAC_OPTIONS["success_prob"] = 0.95

##################################################
###### Line to point reconstruction options ######
##################################################
# Chelani et al. (single-peak finding) options
REFINE_ITER = 3
# Structure Recovery
ESTIMATOR = ["SPF"]

# Ratio of swapping
# SWAP_RATIO = [1.0,0.75,0.5,0.25,0]
SWAP_RATIO = [0]

##################################################
################# Image recovery #################
##################################################
# Input atrribute choices=['depth','depth_sift','depth_rgb','depth_sift_rgb']
INPUT_ATTR = 'depth_sift_rgb'
# INPUT_ATTR = 'depth' # No feature test
# Scale choices=[256,394,512]
SCALE_SIZE = 512
# Crop choices=[256,512]
CROP_SIZE = 512

# Sample size must be bigger than 0
SAMPLE_SIZE = 32
SAMPLE_SIZE = max(SAMPLE_SIZE,1)

##################################################
################## Raise errors ##################
##################################################
# Dataset check
ENERGY_DATASET = ["apt1_kitchen", "apt1_living", "apt2_bed", "apt2_kitchen", 
                    "apt2_living", "apt2_luke", "office1_gates362", "office1_gates381",
                    "office1_lounge", "office1_manolis", "office2_5a", "office2_5b"]
SEVEN_DATASET = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]


# SPARSITY LEVEL should have at least one value for any process to run
def raise_errors(dataset):
    work_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Scale size should be smaller than crop size
    if SCALE_SIZE < CROP_SIZE:
        raise ValueError('SCALE_SIZE must be >= CROP_SIZE')
    
    # Query image & txt file list accordance check
    print("Check query image list accord with query textfile")
    basepath = os.path.join(work_dir, DATASET_MOUNT, getDatasetName(dataset),dataset)
    path_querytxt = os.path.join(basepath,'query_imgs.txt')

# Depth camera parameters for 7scenes and 12scenes dataset
def load_depthcam_parameters(dataset):
    if dataset == '7scenes':
        img_center_x = 320
        img_center_y = 240
        focal_length = 525
        
    elif dataset == '12scenes':
        img_center_x = 320
        img_center_y = 240
        focal_length = 572
    
    return focal_length, img_center_x, img_center_y

def getDatasetName(dataset):
    
    if dataset.lower() in ENERGY_DATASET:
        return "12scenes"

    elif dataset.lower() in SEVEN_DATASET:
        return "7scenes"
    
    else:
        return "none"

def getScale(dataset):
    ### The dataset is already scaled!
    return 1
