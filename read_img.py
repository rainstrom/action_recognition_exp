import os
import cv2
import numpy as np

cur_path = os.path.dirname(__file__)
if cur_path == "":
	cur_path = "."
print cur_path

def read_img_list(lst_fn):
	lst = []
	with open(lst_fn, 'r') as f:
		lines = f.readlines()
		for line in lines:
			splits = line.strip().split(' ')
			lst.append((splits[0],int(splits[1]),int(splits[2])))
	# file_dir, length, label
	return lst

def read_img_iter(lst_fn, num_segments, new_length, prefix, is_mirror, is_crop, crop_size=224):
	assert prefix == "FLOW" or prefix == "RGB" or prefix == "FUSION"
		
	fn_lst = read_img_list(lst_fn)
	all_info = []
	for lst_item in fn_lst:
		vd_dir = lst_item[0]
		vd_length = int(lst_item[1])
		vd_label = int(lst_item[2])
		max_range = (vd_length - new_length) / float(num_segments)
		vd_offsets = [int((i+0.5)*max_range) for i in range(num_segments)]
		
		for vd_offset in vd_offsets:
			if prefix == "RGB":
				img_fn = "image_%04d.jpg" % vd_offset
				img = cv2.imread(vd_dir + '/' + img_fn)
				img = img.transpose((2,0,1))
				yield (img, vd_label)
			elif prefix == "FLOW":
				img_x_fn = "flow_x_%04d.jpg" % vd_offset
				img_y_fn = "flow_y_%04d.jpg" % vd_offset
				img_x = cv2.imread(vd_dir + '/' + img_x_fn)
				img_y = cv2.imread(vd_dir + '/' + img_y_fn)
				img_x = img_x.transpose((2,0,1))
				img_y = img_y.transpose((2,0,1))
				img = np.concatenate((img_x, img_y), axis=0)
				yield (img, vd_label)
			else:
				rgb_img_fn = "image_" + str(vd_offset) + ".jpg"
				rgb_img = cv2.imread(rgb_img_fn)
				rgb_img = rgb_img.transpose((2,0,1))
				flow_img_x_fn = "flow_x_" + str(vd_offset) + ".jpg"
				flow_img_y_fn = "flow_y_" + str(vd_offset) + ".jpg"
				flow_img_x = cv2.imread(flow_img_x_fn)
				flow_img_y = cv2.imread(flow_img_y_fn)
				flow_img_x = flow_img_x.transpose((2,0,1))
				flow_img_y = flow_img_y.transpose((2,0,1))
				flow_img = np.concatenate((flow_img_x, flow_img_y), axis=0)
				yield (rgb_img, flow_img, label)

for img, label in read_img_iter(cur_path + "/dataset_file_examples/val_flow_split1.txt", 25, 10, "FLOW", True, True, 224):
	print img, label
	exit()
