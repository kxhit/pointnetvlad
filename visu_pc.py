from loading_pointclouds import *
import pcl
import open3d as o3d


bin_file = '/media/data/Ford/submap/02/001050.bin'
# bin_file = '/media/work/3D/PointNetVLAD/pointnetvlad/submap_generation/Ford/submap.bin'
pc = np.fromfile(bin_file, dtype=np.float32).reshape(-1,4)[:,:3]
pcd = o3d.PointCloud()
pcd.points = o3d.Vector3dVector(pc)
o3d.draw_geometries([pcd])

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pc)
# o3d.visualization.draw_geometries([pcd])

# pc = load_pc_file(bin_file)
# pc = pc.astype(dtype=np.float32)
# pcd = pcl.PointCloud(pc, fields=('x', 'y', 'z'))

# pcd_file = 'test.pcd'
# pcl.io.savepcd(pcd_file, pcd)