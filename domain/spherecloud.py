import random
import os
from .master import Master
import numpy as np
import poselib
import time
import open3d as o3d
import test_module.recontest as recontest

import utils.pose.pose_estimation as pe
from utils.pose import line
from utils.l2precon import calculate
from static import variable
from utils.pose.vector import *
from utils.colmap.read_write_model import *
from scipy.spatial.transform import Rotation as R
from PIL import Image

np.random.seed(variable.RANDOM_SEED)
random.seed(variable.RANDOM_SEED)

class Spherecloud(Master):
    def __init__(self, dataset_path, output_path):

        # COLMAP 3D Point ID -> Line direction vector mapping
        self.pts_to_line = dict()
        
        # COLMAP 3D point ID -> Center point mapping
        self.pts_to_center = dict()

        # Line index -> COLMAP 3D Point ID mapping
        self.line_to_pts = dict()

        # 3D lines direction (unit vectors)
        self.line_3d = None
  
        # COLMAP info
        self.pts_2d_query = None  # Images.txt
        self.pts_3d_query = None  # Points3D.txt
        self.camera_dict_gt = None  # cameras.txt
        
        # Query image info
        self.queryIds = None
        self.queryNames = None
        self.image_dict_gt = None

        self.resultPose = list() # Estimated poses
        self.resultRecon = list() # Reconstruction results
        self.map_type = "Spherecloud"
        
        self.points_3D_recon = list() # Reconstructed 3D points
        self.lines_3D_recon = list() # Reconstructed 3D lines

        super().__init__(dataset_path, output_path)
        
        # Ids of 3D points from COLMAP
        self.pts_3d_ids = list(self.pts_3d_query.keys())

        # Shuffle 3D points
        np.random.shuffle(self.pts_3d_ids)
            
    def makeLineCloud(self):

        print("Sphere cloud: Line between 3D points and centroid")
        _pts_3d = np.array([v.xyz for v in self.pts_3d_query.values()]) # Get (x,y,z) of 3D points
        _pts_ids = np.array([k for k in self.pts_3d_query.keys()]) # Get COLMAP indices of 3D points

        print('Set a center point as centroid')            
        self.centroid_pt = np.mean(_pts_3d, axis=0)
        _pts_3d -= self.centroid_pt
        
        self.center_pt = np.array([0.0, 0.0, 0.0])        
        self.points_3D, self.line_3d, self.ind_to_id, self.id_to_ind = line.drawlines_ray_clouds_single(_pts_3d, _pts_ids, self.center_pt)

        for i, key in enumerate(self.pts_3d_query.keys()):
            self.pts_to_line[key] = self.line_3d[i] # 3D point COLMAP IDs -> 3D Line unit direction vectors
            self.line_to_pts[i] = key # 3D Line direction vectors index -> 3D point COLMAP IDs
            
    def maskSparsity(self, sparisty_level):
        new_len = int(len(self.pts_3d_ids) * sparisty_level)
        self.sparse_line_3d_ids = set(self.pts_3d_ids[:new_len])

    def matchCorrespondences(self, query_id):

        # Get stored 2D-3D correspondences in query image with COLMAP's database
        connected_pts3d_idx = np.where(self.pts_2d_query[query_id].point3D_ids != -1)[0]
        connected_pts3d_ids = self.pts_2d_query[query_id].point3D_ids[connected_pts3d_idx]
        
        p2 = np.array([self.pts_3d_query[k].xyz for k in connected_pts3d_ids], dtype=np.float64) # 3D points (x,y,z) in COLMAP
        x1 = np.array(self.pts_2d_query[query_id].xys[connected_pts3d_idx], dtype=np.float64) # 2D key points (x,y) in COLMAP

        # Substarct centroid
        if len(p2) > 0:
            p2 -= self.centroid_pt
            
        # Mapping Connected 3D points COLMAP IDs -> point indices!
        pts_to_ind = {}
        for _i, k in enumerate(connected_pts3d_ids):
            pts_to_ind[k] = _i

        # Get valid indices of lines observed in query image (2D points - 3D Lines correspondences) with sparsed lines
        self.valid_pts_3d_ids = self.sparse_line_3d_ids.intersection(set(connected_pts3d_ids))

        newIndex = [] # Point(line) index only in valid 3D pts (lines)
        rays3D = [] # Direction vector of line in valid lines

        for _pid in self.valid_pts_3d_ids:
            newIndex.append(pts_to_ind[_pid])
            rays3D.append(self.pts_to_line[_pid])

        # newIndex가 empty하지 않으면 실행
        if newIndex:
            newIndex = np.array(newIndex)
            # p1: 2D Point
            # x1: 2D Ray
            # p2: 3D Point from query images (RGB-D)
            self._x1 = x1[newIndex]
            self._p2 = p2[newIndex]
            rays3D = -np.asarray(rays3D) # Convert to center -> point direction
        else:
            self._x1 = np.array([])
            self._p2 = np.array([])

        if len(self._x1) > 3:
            # Load depth image corresponding to query imgae        
            gt_img = pe.get_GT_image(query_id, self.pts_2d_query, self.image_dict_gt)
            cam_id = gt_img.camera_id          
            cam_info = pe.convert_cam(self.camera_dict_gt[cam_id])  

            # Convert world 3D points into cam 3D points (GT case)
            q_gt = gt_img.qvec
            R_gt = convert_to_matrix(q_gt)
            t_gt = gt_img.tvec
            self.R_gt = R_gt
            self.t_gt = t_gt
            
            # Make pts3d_RGBD_gt (COLMAP 3D points into Cam coordinate)
            print('Using pseudo-GT RGB-D points')
            pts3d_RGBD_gt = (R_gt @ self._p2.T).T + np.tile(t_gt, [len(self._p2), 1])
            
            ## Recalib as the center of cooridnate moved to centroid
            self.R_gt = R_gt
            
            # Camera center 
            c_gt = -R_gt.T @ t_gt
            c_gt_centroid = c_gt - self.centroid_pt
            
            # Assign new translation
            self.t_gt = -R_gt @ c_gt_centroid
            
            # Assign new RGBD-GT with updated GT pose
            pts3d_RGBD_gt = (self.R_gt @ self._p2.T).T + np.tile(self.t_gt , [len(self._p2), 1])
                                    
            if variable.getDatasetName(self.dataset) == '12scenes':
                depth_img, self.no_depth_file = self.load_depth_img(gt_img.name.split('/')[1])
            
            elif variable.getDatasetName(self.dataset) == '7scenes':    
                depth_img, self.no_depth_file = self.load_depth_img(gt_img.name.replace('/', '_'))
            
            # If no depth skip..
            if self.no_depth_file == False:
                        
                # Make 3D point cloud from depth imgae
                print('Extract 3D key points!')
                pts3d_RGBD, valid_indices = self.make_pts3d_RGBD(points_2D=self._x1, depth=depth_img, cam_info=cam_info, z_GT=pts3d_RGBD_gt[:,2])
                
                # Get only valid
                self.pts3d_RGBD = np.array(pts3d_RGBD[valid_indices], dtype=np.float64) 
                self.pts3d_COLMAP = np.array(self._p2[valid_indices], dtype=np.float64)
                self.points2D_query = np.array(self._x1[valid_indices], dtype=np.float64)
                self.rays3D = np.array(rays3D[valid_indices], dtype=np.float64)

                # Check data
                print('Found correspondences:', len(self.pts3d_RGBD))
                assert len(self.pts3d_RGBD) == len(self.pts3d_COLMAP) == len(self.points2D_query) == len(self.rays3D)
            else:
                print('No depth file!')
        
    def addNoise(self, noise_level):
        super().addNoise(noise_level)

    def estimatePose(self, query_id):
        
        # If Depth file doesn;t exist
        if self.no_depth_file == True:
            print('Depth file doesnt exist')
            exit(-1)
            # return 0

        # If there's more than 3 points
        if self._x1.shape[0] >= 3:
            gt_img = pe.get_GT_image(query_id, self.pts_2d_query, self.image_dict_gt)
            cam_id = gt_img.camera_id          
            cam_dict = pe.convert_cam(self.camera_dict_gt[cam_id])
            print('gt_img.name:', gt_img.name)

            lambda1 = variable.LAMBDA1
            lambda2 = variable.LAMBDA2
            
            start = time.time()            
            ### Use RGBD 3D points
            res = poselib.estimate_vp3p_absolute_pose(self.pts3d_COLMAP, self.rays3D, self.points2D_query, self.pts3d_RGBD, [cam_dict], lambda1, lambda2, variable.RANSAC_OPTIONS, variable.BUNDLE_OPTIONS, variable.REFINE_OPTION)
            end = time.time()
                
            # Save pose acc
            super().savePoseAccuracy(res, gt_img, cam_dict, (end-start))

        else:
            print("TOO sparse point cloud")

    def savePose(self, sparisty_level, noise_level):
        super().savePose(sparisty_level, noise_level)

    def saveAllPoseCSV(self):
        super().saveAllPoseCSV()

    def recoverPts(self, estimator, sparsity_level, noise_level):
        
        print("Sphere Clouds recover 3D points", "\n")

        self.sparse_pts_3d_ids = []
        self.id_to_ind_recon = {}
        self.ind_to_id_recon = {}
        self.points_3D_recon = []
        self.lines_3D_recon = []

        for i, _pts_3d_id in enumerate(self.sparse_line_3d_ids):
            self.sparse_pts_3d_ids.append(_pts_3d_id)
            self.points_3D_recon.append(self.pts_3d_query[_pts_3d_id].xyz)
            self.lines_3D_recon.append(self.pts_to_line[_pts_3d_id])

            # Real 3D points
            self.id_to_ind_recon[_pts_3d_id] = i
            self.ind_to_id_recon[i] = _pts_3d_id

        self.points_3D_recon = np.array(self.points_3D_recon)
        self.lines_3D_recon = np.array(self.lines_3D_recon)

        ref_iter = variable.REFINE_ITER
        
        if estimator == 'SPF':
            ests, ests_pts = calculate.coarse_est_spf(self.points_3D_recon, self.lines_3D_recon)
            ests_pts = calculate.refine_est_spf(self.points_3D_recon, self.lines_3D_recon, ests, ref_iter)
            info = [sparsity_level, noise_level, 0.0, estimator]
            super().saveReconpoints(ests_pts, info)
        
        elif estimator=='TPF':
            print("Sphere cloud should't be estimated with TPF")
            pass

    def reconTest(self, estimator):
        # reconTest
        recontest.recontestPTidx([self.points_3D_recon], [self.ind_to_id_recon], self.pts_3d_query)

    def test(self, recover, esttype):
        # recon test
        print("Consistency test for", self.map_type)
        if recover:
            self.reconTest(esttype)
            
    def load_depth_img(self, query_name):
        
        # Set flag
        no_depth_file = False

        if variable.RESOURCE != 'local':
            NAS_path = '/workspace/mnt' # Docker environment
        elif variable.RESOURCE == 'local':
            NAS_path = '/home/moon/Desktop/mnt' # Local
            # NAS_path = '/home/moon/Desktop/single_raycloud/dataset' # Local

        dataset_path = os.path.join(NAS_path, variable.getDatasetName(self.dataset), self.dataset)
        print('dataset_path:', dataset_path)

        ## Load raw depth
        folder_path = os.path.join(dataset_path, 'raw_depth')
        if variable.getDatasetName(self.dataset) == '7scenes':
            depth_img_name_tmp = query_name.replace('color.png', 'depth.png')
            depth_img_name = depth_img_name_tmp.replace('/', '_')
            print('depth_img_name:', depth_img_name)
            
        elif variable.getDatasetName(self.dataset) == '12scenes':    
            depth_img_name_tmp = query_name.replace('color.jpg', 'depth.png')
            depth_img_name = depth_img_name_tmp
            
        dataset_path = os.path.join(NAS_path, variable.getDatasetName(self.dataset), self.dataset)
        depth_path = os.path.join(folder_path, depth_img_name)
        if not os.path.exists(depth_path):
            no_depth_file = True
            return None, no_depth_file
                
        depth_img = Image.open(depth_path)
        depth_img = np.asarray(depth_img, dtype=np.float64)
        
        # convert to meter scale
        scaling_factor = 1000
        depth_img /= scaling_factor 
            
        return depth_img, no_depth_file
    
    def make_pts3d_RGBD(self, points_2D, depth, cam_info, z_GT):

        pts3d_RGBD = []
        inlier_points3D_indices = [] # Inlier indices
        
        if variable.getDatasetName(self.dataset) == '12scenes':
            
            # Load RGB camera parameter
            # PINHOLE model -> fx, fy, cx, cy
            focal_length_x, focal_length_y, img_center_x, img_center_y = cam_info['params'][0], cam_info['params'][1], cam_info['params'][2], cam_info['params'][3]
            
            self.focal_length_x = focal_length_x
            self.focal_length_y = focal_length_y
            self.img_center_x = img_center_x
            self.img_center_y = img_center_y
            
            # Load Depth camera parameter
            depth_focal_length = 572 # fx = fy
            depth_img_center_x = 320
            depth_img_center_y = 240
            
            for idx, kpt in enumerate(points_2D):
                '''
                v -> height
                u -> width
                '''

                # RGB keypoint -> to normalized plane -> depth image plane
                u = (kpt[0] - img_center_x) / focal_length_x * depth_focal_length
                v = (kpt[1] - img_center_y) / focal_length_y * depth_focal_length

                # Add principal point
                u += depth_img_center_x
                v += depth_img_center_y
                
                ## Use GT depth
                if variable.USE_DEPTH_ORACLE:
                    z = z_GT[idx] 
                ## Use depth measurement
                else:
                    z_neighbor =[]
                    u = round(np.clip(u, a_min=0, a_max=639))
                    v = round(np.clip(v, a_min=0, a_max=479))
                    u_left = np.clip(u-1, a_min=0, a_max=639)
                    u_right = np.clip(u+1, a_min=0, a_max=639)
                    v_up = np.clip(v-1, a_min=0, a_max=479)
                    v_down = np.clip(v+1, a_min=0, a_max=479)
                    
                    z_orig = depth[v][u]
                    z_left = depth[v][u_left]
                    z_right = depth[v][u_right]
                    z_up = depth[v_up][u]
                    z_down = depth[v_down][u]
                    
                    z_neighbor.append(z_orig)
                    z_neighbor.append(z_left)
                    z_neighbor.append(z_right)
                    z_neighbor.append(z_up)
                    z_neighbor.append(z_down)
                    
                    # z neighbor inlier average (depth ==0 is outlier)
                    z_neighbor_sum =0
                    z_neighbor_avg =0
                    z_inlier=0
                    for z_neigh in z_neighbor:
                        if(z_neigh==0):
                            continue
                        z_inlier +=1
                        z_neighbor_sum+=z_neigh
                    
                    if(z_inlier!=0):
                        z = z_neighbor_sum/float(z_inlier)
                    else:
                        z=0

                # Get depth value only in valid range
                if variable.DEPTH == 'RAW':
                    if z > 0.01 and z <= 5.0: # depth valid range
                        inlier_points3D_indices.append(idx)
                elif variable.DEPTH == 'RENDERED':
                    inlier_points3D_indices.append(idx)
                                    
                # Use RGB keypoints
                x = (kpt[0] - img_center_x) / focal_length_x * z
                y = (kpt[1] - img_center_y) / focal_length_y * z

                pts3d_RGBD.append([x, y, z])
            
        elif variable.getDatasetName(self.dataset) == '7scenes':
            
            rgb_focal_length = 525
            depth_focal_length = 525
            img_center_x = 320
            img_center_y = 240
            
            self.focal_length = rgb_focal_length
            self.img_center_x = img_center_x
            self.img_center_y = img_center_y
            
            if variable.PGT_TYPE == 'sfm_gt':        
                # SimpleRadial; f, cx, cy, k
                focal_length_x, focal_length_y, img_center_x, img_center_y = cam_info['params'][0], cam_info['params'][0], cam_info['params'][1], cam_info['params'][2]
            else:
                rgb_focal_length, img_center_x, img_center_y = cam_info['params'][0], cam_info['params'][1], cam_info['params'][2]
            
            for idx, kpt in enumerate(points_2D):
                '''
                v -> height
                u -> width
                '''
                u = kpt[0]
                v = kpt[1]

                ## Use GT depth
                if variable.USE_DEPTH_ORACLE:
                    z = z_GT[idx]
                ## Use depth measurement
                else:
                    z_neighbor =[]
                    u = round(np.clip(u, a_min=0, a_max=639))
                    v = round(np.clip(v, a_min=0, a_max=479))
                    u_left = np.clip(u-1, a_min=0, a_max=639)
                    u_right = np.clip(u+1, a_min=0, a_max=639)
                    v_up = np.clip(v-1, a_min=0, a_max=479)
                    v_down = np.clip(v+1, a_min=0, a_max=479)
                    z_orig = depth[v][u]
                    z_left = depth[v][u_left]
                    z_right = depth[v][u_right]
                    z_up = depth[v_up][u]
                    z_down = depth[v_down][u]
                    z_neighbor.append(z_orig)
                    z_neighbor.append(z_left)
                    z_neighbor.append(z_right)
                    z_neighbor.append(z_up)
                    z_neighbor.append(z_down)
                    # z neighbor inlier average (depth ==0 is outlier)
                    z_neighbor_sum =0
                    z_neighbor_avg =0
                    z_inlier=0
                    for z_neigh in z_neighbor:
                        if(z_neigh==0):
                            continue
                        z_inlier +=1
                        z_neighbor_sum+=z_neigh
                    if(z_inlier!=0):
                        z = z_neighbor_sum/float(z_inlier)
                    else:
                        z=0
                    
                if variable.DEPTH == 'RAW':
                    if z >= 0.01 and z <= 5.0:  # depth valid range: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6912003 
                        inlier_points3D_indices.append(idx)
                elif variable.DEPTH == 'RENDERED':
                    inlier_points3D_indices.append(idx)
                
                x = (kpt[0] - img_center_x) * z / rgb_focal_length
                y = (kpt[1] - img_center_y) * z / rgb_focal_length

                pts3d_RGBD.append([x, y, z])

        pts3d_RGBD = np.asarray(pts3d_RGBD, dtype=np.float64)
        inlier_points3D_indices = np.asarray(inlier_points3D_indices, dtype=np.int32)

        return pts3d_RGBD, inlier_points3D_indices