from loading_pointclouds import *
import pcl

bin_file = '/media/work/3D/PointNetVLAD/benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap/1400505893170765.bin'
pc = load_pc_file(bin_file)
pc = pc.astype(dtype=np.float32)
pcd = pcl.PointCloud(pc, fields=('x', 'y', 'z'))
pcd_file = 'test.pcd'
pcl.io.savepcd(pcd_file, pcd)