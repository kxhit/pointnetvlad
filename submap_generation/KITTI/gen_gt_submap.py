"""
使用gt pose 拼前后 dist_r m 的激光点云作为submap,实际slam系统
中可以用
"""

import numpy as np
import os
# import pcl
# import pclpy
from pypcd import pypcd
import pykitti
from tqdm import tqdm
from gen_gt import *

dist_r = 10 # 10m submap

# def get_submap_ids(id, dist_r, dataset):
#     dataset.pose


# Change this to the directory where you store KITTI data
basedir = '/media/data/kitti/odometry/dataset'


# Specify the dataset to load
# sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
sequences = ['08']
# for sequence
for sequence in tqdm(sequences):
    # Load the data. Optionally, specify the frame range to load.
    dataset = pykitti.odometry(basedir, sequence)
    # Output dir
    outputdir = '/media/work/data/kitti/odometry/submap'
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    outputdir = outputdir + '/' + sequence
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    last_id_begin = 0
    last_id_end = 0
    for frame_id in tqdm(range(len(dataset.timestamps))):
        # get submap frame ids
        cur_pose = dataset.poses[frame_id]

        # find begin_id which distance is just less than dist_r
        while(True):
            begin_pose = dataset.poses[last_id_begin]
            if p_dist(cur_pose, begin_pose, threshold=dist_r): # return true if dist < thresh
                break
            else:
                last_id_begin += 1
        # find end_id which distance is just larger than dist_r
        while (True):
            end_pose = dataset.poses[last_id_end]
            if p_dist(cur_pose, end_pose, threshold=dist_r): # return true if dist < thresh
                if last_id_end == len(dataset.timestamps) - 1:
                    break
                else:
                    last_id_end += 1
            else:
                break

        # submap range [begin_id, end_id]
        print("submap range ", last_id_begin, "~", last_id_end)
        # RT transform
        submap_tmp = dataset.get_velo(frame_id)
        Tr_velo_to_cam = dataset.calib.T_cam0_velo
        cur_pose = np.dot(cur_pose, Tr_velo_to_cam)
        #TODO: too slow
        for sub_id in range(last_id_begin, last_id_end+1):
            # get lidar points
            lidar = dataset.get_velo(sub_id) # Nx4
            # # downsample
            # num = lidar.shape[0]
            # index = np.random.choice(np.arange(num), int(0.5*num),replace=False).reshape(-1)
            # lidar = lidar[index]

            intensity = lidar[:, -1].copy()
            lidar[:, -1] = 1

            # get pose
            pose = dataset.poses[sub_id]    # 4x4
            pose = np.dot(pose, Tr_velo_to_cam)

            lidar_in_cur = np.dot(lidar, np.dot(np.linalg.inv(cur_pose), pose).T)

            lidar_in_cur[:, -1] = intensity
            lidar_in_cur = lidar_in_cur.astype(dtype=np.float32)
            submap_tmp = np.concatenate((submap_tmp, lidar_in_cur), axis=0)

        # save submaps as pcd files
        # submap = pcl.PointCloud(submap_tmp, ('x', 'y', 'z', 'intensity'))
        submap = pypcd.make_xyz_label_point_cloud(submap_tmp)
        # submap = pypcd.PointCloud({'x', 'y', 'z', 'intensity'},submap_tmp)

        pcd_file = outputdir + '/' + '%06d'%frame_id + '.pcd'
        # pcl.io.savepcd(pcd_file, submap)
        submap.save_pcd(pcd_file)
        # pcl.load()
        # pc_read = pcl.io.loadpcd(pcd_file)

        # assert pc_read == submap