import sys
import os
sys.path.insert(0,'../')

from full_test import full_test

caffemodel = './flow_10img_iter_25000.caffemodel'
net_prototxt = './train_val_fast.prototxt'
pkl_dir = './'
experiment_name = 'flow_10img_iter25000'
output_layer_names = ['flow_fc8']
device_id = 0
do_mirror = False
do_crop = False
test_segments = 1
test_batch_size = 1

os.chdir('flow_10img')
full_test(caffemodel, net_prototxt, pkl_dir, experiment_name, output_layer_names, device_id, do_mirror, do_crop, test_segments, test_batch_size)
