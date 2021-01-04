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
# from gen_gt import *
from liegroups.numpy import SO3
# import open3d as o3d


def listDir(path, list_name):
    """
    :param path: root_dir
    :param list_name: abs paths of all files under the root_dir
    :return:
    """
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listDir(file_path, list_name)
        else:
            list_name.append(file_path)

dist_r = 10 # 10m submap

# Change this to the directory where you store Ford data
basedir = '/media/work/data/Ford/'

# T_bl6D = [2.4,-0.01,-2.3,np.pi,0,np.pi/2]   # velodyne in body frame, parameter of X_bl in paper Eq.1

# Specify the dataset to load
# sequences = ['01','02']
sequences = ['01']
# for sequence
for sequence in tqdm(sequences):
    # Load the data. Optionally, specify the frame range to load.
    # dataset = pykitti.odometry(basedir, sequence)
    # load ford dataset pcs poses
    dataset_dir = os.path.join(basedir,'IJRR-Dataset-' + str(int(sequence)))
    poses6D = np.fromfile(os.path.join(dataset_dir, "SG/poses_ws.bin"), dtype=np.float32).reshape(-1, 6)
    # 6-DoF -> 4x4
    poses = np.zeros((poses6D.shape[0], 4, 4))
    poses[:, 3, 3] = 1
    for i in range(poses.shape[0]):
        poses[i, :3, :3] = SO3.from_rpy(poses6D[i, 3], poses6D[i, 4], poses6D[i, 5]).mat
        poses[i, :3, 3] = poses6D[i, :3]
    # T_bl = np.zeros((4, 4))
    # T_bl[3, 3] = 1
    # T_bl[:3, :3] = SO3.from_rpy(T_bl6D[3],T_bl6D[4],T_bl6D[5]).mat
    # T_bl[:3, 3] = T_bl6D[:3]

    scan_files = []

    listDir(os.path.join(dataset_dir, 'SCANS_bin_32'), scan_files)
    scan_files.sort()
    # Output dir
    outputdir = '/media/data/Ford/submap'


    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    outputdir = outputdir + '/' + sequence
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    last_id_begin = 0
    last_id_end = 0
    frame_nums = poses.shape[0]
    for frame_id in tqdm(range(frame_nums)):
        # get submap frame ids
        cur_pose = poses6D[frame_id]

        # find begin_id which distance is just less than dist_r
        while(True):
            begin_pose = poses6D[last_id_begin]
            if np.linalg.norm(cur_pose[0:3] - begin_pose[0:3]) < dist_r:
                 # return true if dist < thresh
                # print(np.linalg.norm(cur_pose[0:3] - begin_pose[0:3]))
                break
            else:
                last_id_begin += 1
        # find end_id which distance is just larger than dist_r
        while (True):
            end_pose = poses6D[last_id_end]
            if np.linalg.norm(cur_pose[0:3] - end_pose[0:3]) < dist_r: # return true if dist < thresh
                if last_id_end == frame_nums - 1:
                    break
                else:
                    last_id_end += 1
            else:
                break

        # submap range [begin_id, end_id]
        print("submap range ", last_id_begin, "~", last_id_end)
        # RT transform
        # submap_tmp = dataset.get_velo(frame_id)
        submap_tmp = np.fromfile(scan_files[frame_id],dtype=np.float32).reshape(-1,4)
        submap_tmp = submap_tmp[np.argwhere(submap_tmp[:, 2] > -2.2).reshape(-1), :]    # remove ground
        ind = np.random.choice(submap_tmp.shape[0], 4096, replace=False)    # down sample
        submap_tmp = submap_tmp[ind, :]
        # Tr_velo_to_cam = dataset.calib.T_cam0_velo #
        # cur_pose = np.dot(cur_pose, Tr_velo_to_cam)
        # cur_pose = np.dot(poses[frame_id], T_bl)
        cur_pose = poses[frame_id]
        #TODO: too slow
        for sub_id in range(last_id_begin, last_id_end+1):
            # get lidar points
            # lidar = dataset.get_velo(sub_id) # Nx4
            lidar = np.fromfile(scan_files[sub_id],dtype=np.float32).reshape(-1,4)
            lidar = lidar[np.argwhere(lidar[:, 2] > -2.2).reshape(-1),:]    # remove ground
            ind = np.random.choice(lidar.shape[0], 4096, replace=False)
            lidar = lidar[ind, :] # down sample
            # # downsample
            # num = lidar.shape[0]
            # index = np.random.choice(np.arange(num), int(0.5*num),replace=False).reshape(-1)
            # lidar = lidar[index]

            intensity = lidar[:, -1].copy()
            lidar[:, -1] = 1

            # get pose
            # pose = np.dot(poses[sub_id],T_bl)    # 4x4
            pose = poses[sub_id]
            # pose = np.dot(pose, Tr_velo_to_cam)

            # lidar_in_cur = np.dot(lidar, np.dot(np.linalg.inv(cur_pose), pose).T)
            lidar_in_cur = np.dot(np.dot(np.linalg.inv(cur_pose),pose),lidar.T).T
            # lidar_in_cur = np.dot(np.dot(cur_pose, np.linalg.inv(pose)), lidar.T).T

            lidar_in_cur[:, -1] = intensity
            lidar_in_cur = lidar_in_cur.astype(dtype=np.float32)
            submap_tmp = np.concatenate((submap_tmp, lidar_in_cur), axis=0)



        # save submaps as pcd files
        # submap = pcl.PointCloud(submap_tmp, ('x', 'y', 'z', 'intensity'))
        # submap = pypcd.make_xyz_label_point_cloud(submap_tmp)
        # submap = pypcd.PointCloud({'x', 'y', 'z', 'intensity'},submap_tmp) ##
        #
        # pcd_file = outputdir + '/' + '%06d'%frame_id + '.pcd'
        # # pcl.io.savepcd(pcd_file, submap)
        # submap.save_pcd(pcd_file)
        # pcl.load()
        # pc_read = pcl.io.loadpcd(pcd_file)

        # assert pc_read == submap
        # -25m~25m
        l = 25
        pc = submap_tmp
        ind = np.argwhere(pc[:, 0] <= l).reshape(-1)
        pc = pc[ind]
        ind = np.argwhere(pc[:, 0] >= -l).reshape(-1)
        pc = pc[ind]
        ind = np.argwhere(pc[:, 1] <= l).reshape(-1)
        pc = pc[ind]
        ind = np.argwhere(pc[:, 1] >= -l).reshape(-1)
        pc = pc[ind]
        ind = np.argwhere(pc[:, 2] <= l).reshape(-1)
        pc = pc[ind]
        ind = np.argwhere(pc[:, 2] >= -l).reshape(-1)
        pc = pc[ind]
        # save as .bin files
        bin_file = outputdir + '/' + '%06d' % frame_id + '.bin'

        pc.tofile(bin_file)
