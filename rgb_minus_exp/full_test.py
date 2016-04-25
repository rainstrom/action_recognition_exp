from ../full_test import full_test


caffemodel = './cuhk_action_recognition_vgg_16_conv1minus_rgb5_initfromvgg_split1_rgb_iter_20000.caffemodel'
net_prototxt = 'conv1minus_rgb5/train_val_fast.prototxt'
pkl_dir = "conv1minus_rgb5/"
experiment_name = "conv1minus_rgb5_initfromvgg_iter20000"
output_layer_names = ["fc8"]
device_id = 3

do_mirror = False
do_crop = False
test_segments = 1
test_batch_size = 1 

full_test(caffemodel, net_prototxt, pkl_dir, experiment_name, output_layer_names, device_id, do_mirror, do_crop, test_segments, test_batch_size)