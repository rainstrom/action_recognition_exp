import sys
from full_test import full_test
sys.path.insert(0, '../../')

device_id = 2

caffemodel = './flow_5img_iter_15000.caffemodel'
net_prototxt = './train_val_fast.prototxt'
pkl_dir = "./"
experiment_name = "flow_5img_iter15000"
output_layer_names = ["flow_fc8"]

do_mirror = True
do_crop = True
test_segments = 25
test_batch_size = 5

full_test(caffemodel, net_prototxt, pkl_dir, experiment_name, output_layer_names, device_id, do_mirror, do_crop, test_segments, test_batch_size)
