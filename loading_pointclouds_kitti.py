import os
import pickle
import numpy as np
import random
import json

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_PATH=os.path.join(BASE_DIR, '../benchmark_datasets/')
# print(BASE_PATH)
#BASE_PATH = "/media/deep-three/Deep_Store/CVPR2018/benchmark_datasets/"
dict1 = "./KITTI_all/positive_sequence_D-3_T-0.json"
f = open(dict1, "r")
DICT_1 = json.load(f)

dict2 = "./KITTI_all/positive_sequence_D-20_T-0.json"
f = open(dict2, "r")
DICT_2 = json.load(f)

seq_len = {"00":4541, "01":1101, "02":4661, "03":801, "04":271, "05":2761,
                        "06":1101,"07":1101,"08":4071,"09":1591,"10":1201}

def rotate_pc(pc, axis="y", max_angle=10):
    """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    # rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    assert pc.shape[1] == 3
    rotation_angle = np.random.uniform() * 2 * np.pi * max_angle/360
    # rotation_angle = 1 * 2 * np.pi * max_angle/360

    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)

    if axis == "y":
        # along y pitch
        rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])

    elif axis == "x":
        # along x roll
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
    elif axis == "z":
        # along z yaw
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
    else:
        print("axis wrong: ", axis)
        exit(-1)
    rotated_data = np.dot(pc, rotation_matrix)
    return rotated_data

def random_drop(points, drop_angle=30):
    '''
	:param points: Nx3
	:param drop_angle:
	:return:
	'''

    # randomly drop some points
    start_angle = np.random.random()
    start_angle *= 360

    end_angle = (start_angle + drop_angle) % 360

    angle = np.arctan2(points[:, 1], points[:, 0])
    angle = angle * 180 / np.pi
    angle += 180
    # print("angle:", angle)
    if end_angle > start_angle:
        remain_id = np.argwhere(angle < start_angle).reshape(-1)
        remain_id = np.append(remain_id, np.argwhere(angle > end_angle).reshape(-1))
    else:
        remain_id = np.argwhere((angle > end_angle) & (angle < start_angle)).reshape(-1)
    # print(remain_id)
    return points[remain_id,:]

def fov100(points):
    '''
	:param points: Nx3
	:param drop_angle:
	:return:
	'''

    start_angle = 130
    end_angle = 230
    angle = np.arctan2(points[:, 1], points[:, 0])
    angle = angle * 180 / np.pi
    angle += 180
    p1 = start_angle < angle
    p2 = angle < end_angle
    p = p1 & p2
    remain_id = np.argwhere(p).reshape(-1)
    return points[remain_id,:]

def load_pc_file(filename):
    # returns Nx3 matrix
    pc = np.fromfile(filename, dtype=np.float32).reshape(-1,4)[:,:3] # xyz
    # # todo fov 100
    # pc = fov100(pc)
    # # # todo random drop30    drop90
    # pc = random_drop(pc, 30)
    # preprocess as paper
    # -25~25 cubic
    l = 25
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
    # sample to 4096
    if pc.shape[0] >= 4096:
        ind = np.random.choice(pc.shape[0], 4096, replace=False)
        pc = pc[ind, :]
    else:
        ind = np.random.choice(pc.shape[0], 4096, replace=True)
        pc = pc[ind, :]
    # rescale to [-1,1] with zero mean
    mean = np.mean(pc, axis=0)
    pc = pc - mean
    scale = np.max(abs(pc))
    pc = pc/scale

    # todo drop0_rotate
    pc = rotate_pc(pc, axis="z", max_angle=360)
    return pc

def load_pc_files(filenames):
    pcs=[]
    for filename in filenames:
        #print(filename)
        pc=load_pc_file(filename)
        assert pc.shape[0]== 4096
        pcs.append(pc)
    pcs=np.array(pcs)
    return pcs

def get_queries_dict(filename):
	#key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("Queries Loaded.")
		return queries

def get_sets_dict(filename):
	#[key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}},key_dataset:{key_pointcloud:{'query':file,'northing':value,'easting':value}}, ...}
	with open(filename, 'rb') as handle:
		trajectories = pickle.load(handle)
		print("Trajectories Loaded.")
		return trajectories

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        #-90 to 90
        rotation_angle = (np.random.uniform()*np.pi)- np.pi/2.0
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.005, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def get_pos(file_name):
    sq = file_name.split('/')[-2]
    index = file_name.split('/')[-1].split('.')[0]
    assert sq in DICT_1.keys()
    # assert sq in DICT_2.keys()
    sq_1 = DICT_1[sq]
    # sq_2 = DICT_2[sq]
    if str(int(index)) in sq_1:
        positives = sq_1[str(int(index))]
    else:
        positives = []
    # print(positives)
    return positives

def get_neg(file_name):
	sq = file_name.split('/')[-2]
	index = file_name.split('/')[-1].split('.')[0]
    # assert sq in DICT_1.keys()
	assert sq in DICT_2.keys()
	# sq_1 = DICT_1[sq]
	sq_2 = DICT_2[sq]
	neg_set = set(np.arange(seq_len[sq])).difference(sq_2[str(int(index))])  # neg_set = all id - dict_2
	negtives = list(neg_set)
	return negtives

def get_query_tuple(query_file, num_pos, num_neg, TRAINING_FILES, hard_neg=[], other_neg=False):
	#get query tuple for dictionary entry
	#return list [query,positives,negatives]
	base_path = query_file[:-11]
	positives_files = get_pos(query_file)
	negtives_files = get_neg(query_file)
	query=load_pc_file(query_file) #Nx3

	# random.shuffle(dict_value["positives"])
	random.shuffle(positives_files)
	pos_files=[]

	for i in range(num_pos):
		# pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
		pos_files.append(os.path.join(base_path, '%06d'%int(positives_files[i])+".bin"))
	#positives= load_pc_files(dict_value["positives"][0:num_pos])
	positives=load_pc_files(pos_files)

	neg_files=[]
	neg_indices=[]
	if(len(hard_neg)==0):
		# dict_value["negatives"] -> negtives
		# random.shuffle(dict_value["negatives"])
		random.shuffle(negtives_files)
		for i in range(num_neg):
			# neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
			neg_files.append(os.path.join(base_path, '%06d'%int(negtives_files[i])+".bin"))
			# neg_indices.append(dict_value["negatives"][i])
			neg_indices.append(os.path.join(base_path, '%06d' % int(negtives_files[i]) + ".bin"))

	else:
		# random.shuffle(dict_value["negatives"])
		random.shuffle(negtives_files)
		for i in hard_neg:
			# neg_files.append(QUERY_DICT[i]["query"])
			# neg_indices.append(i)
			neg_files.append(os.path.join(base_path, '%06d' % int(i) + ".bin"))
			neg_indices.append(os.path.join(base_path, '%06d' % int(i) + ".bin"))
		j=0
		while(len(neg_files)<num_neg):

			# if not dict_value["negatives"][j] in hard_neg:
			if not negtives_files[j] in hard_neg:
				# neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
				# neg_indices.append(dict_value["negatives"][j])
				neg_files.append(os.path.join(base_path, '%06d' % int(negtives_files[j]) + ".bin"))
				neg_indices.append(os.path.join(base_path, '%06d' % int(negtives_files[j]) + ".bin"))
			j+=1

	negatives=load_pc_files(neg_files)

	if(other_neg==False):
		return [query,positives,negatives]
	#For Quadruplet Loss
	else:
		#get neighbors of negatives and query
		neighbors=[]
		# for pos in dict_value["positives"]:
		for pos in positives_files:
			neighbors.append(pos)
		for neg in neg_indices:
			# for pos in QUERY_DICT[neg]["positives"]:
			neg_postives_files = get_pos(neg)
			for pos in neg_postives_files:
				neighbors.append(os.path.join(base_path, '%06d' % int(pos) + ".bin"))
		# possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
		possible_negs = list(set(TRAINING_FILES) - set(neighbors))
		random.shuffle(possible_negs)

		if(len(possible_negs)==0):
			return [query, positives, negatives, np.array([])]

		# neg2= load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
		neg2 = load_pc_file(possible_negs[0])

		return [query,positives,negatives,neg2]


def get_rotated_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[],other_neg=False):
	query=load_pc_file(dict_value["query"]) #Nx3
	q_rot= rotate_point_cloud(np.expand_dims(query, axis=0))
	q_rot= np.squeeze(q_rot)

	random.shuffle(dict_value["positives"])
	pos_files=[]
	for i in range(num_pos):
		pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
	#positives= load_pc_files(dict_value["positives"][0:num_pos])
	positives=load_pc_files(pos_files)
	p_rot= rotate_point_cloud(positives)

	neg_files=[]
	neg_indices=[]
	if(len(hard_neg)==0):
		random.shuffle(dict_value["negatives"])
		for i in range(num_neg):
			neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
			neg_indices.append(dict_value["negatives"][i])
	else:
		random.shuffle(dict_value["negatives"])
		for i in hard_neg:
			neg_files.append(QUERY_DICT[i]["query"])
			neg_indices.append(i)
		j=0
		while(len(neg_files)<num_neg):
			if not dict_value["negatives"][j] in hard_neg:
				neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
				neg_indices.append(dict_value["negatives"][j])
			j+=1
	negatives=load_pc_files(neg_files)
	n_rot=rotate_point_cloud(negatives)

	if(other_neg==False):
		return [q_rot,p_rot,n_rot]

	#For Quadruplet Loss
	else:
		#get neighbors of negatives and query
		neighbors=[]
		for pos in dict_value["positives"]:
			neighbors.append(pos)
		for neg in neg_indices:
			for pos in QUERY_DICT[neg]["positives"]:
				neighbors.append(pos)
		possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
		random.shuffle(possible_negs)

		if(len(possible_negs)==0):
			return [q_jit, p_jit, n_jit, np.array([])]

		neg2= load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
		n2_rot= rotate_point_cloud(np.expand_dims(neg2, axis=0))
		n2_rot= np.squeeze(n2_rot)

		return [q_rot,p_rot,n_rot,n2_rot]

def get_jittered_tuple(dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[],other_neg=False):
	query=load_pc_file(dict_value["query"]) #Nx3
	#q_rot= rotate_point_cloud(np.expand_dims(query, axis=0))
	q_jit= jitter_point_cloud(np.expand_dims(query, axis=0))
	q_jit= np.squeeze(q_jit)

	random.shuffle(dict_value["positives"])
	pos_files=[]
	for i in range(num_pos):
		pos_files.append(QUERY_DICT[dict_value["positives"][i]]["query"])
	#positives= load_pc_files(dict_value["positives"][0:num_pos])
	positives=load_pc_files(pos_files)
	p_jit= jitter_point_cloud(positives)

	neg_files=[]
	neg_indices=[]
	if(len(hard_neg)==0):
		random.shuffle(dict_value["negatives"])
		for i in range(num_neg):
			neg_files.append(QUERY_DICT[dict_value["negatives"][i]]["query"])
			neg_indices.append(dict_value["negatives"][i])
	else:
		random.shuffle(dict_value["negatives"])
		for i in hard_neg:
			neg_files.append(QUERY_DICT[i]["query"])
			neg_indices.append(i)
		j=0
		while(len(neg_files)<num_neg):
			if not dict_value["negatives"][j] in hard_neg:
				neg_files.append(QUERY_DICT[dict_value["negatives"][j]]["query"])
				neg_indices.append(dict_value["negatives"][j])
			j+=1
	negatives=load_pc_files(neg_files)
	n_jit=jitter_point_cloud(negatives)

	if(other_neg==False):
		return [q_jit,p_jit,n_jit]

	#For Quadruplet Loss
	else:
		#get neighbors of negatives and query
		neighbors=[]
		for pos in dict_value["positives"]:
			neighbors.append(pos)
		for neg in neg_indices:
			for pos in QUERY_DICT[neg]["positives"]:
				neighbors.append(pos)
		possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
		random.shuffle(possible_negs)

		if(len(possible_negs)==0):
			return [q_jit, p_jit, n_jit, np.array([])]

		neg2= load_pc_file(QUERY_DICT[possible_negs[0]]["query"])
		n2_jit= jitter_point_cloud(np.expand_dims(neg2, axis=0))
		n2_jit= np.squeeze(n2_jit)

		return [q_jit,p_jit,n_jit,n2_jit]

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


if __name__ == "__main__":
	TRAIN_DICT = get_queries_dict('generating_queries/training_queries_baseline.pickle')