import numpy as np
import caffe
import sys
import cPickle as pickle

caffe.set_device(7)
caffe.set_mode_gpu()
num_videos = 3783
# num_videos = 20

# model to be tested
caffemodel = 'caffemodels/cuhk_action_recognition_vgg_16_conv1minus_rgb5_split1_rgb_iter_10000.caffemodel'
net_prototxt = 'conv1minus_rgb5/vgg_16_rgb5_conv1minus_train_val_fast.prototxt'
experiment_name = "rgb5_conv1minus"
output_layer_name = "fc8"

# parameters to set
do_mirror = False
do_crop = False
test_segments = 25
test_batch_size = 1 

# generate test_param_tag
test_param_tag = ("mirror" if do_mirror else "no_mirror") + ("_crop" if do_crop else "_no_crop") + "_seg" + str(test_segments)
pickle_file = 'full_test_' + experiment_name + '_' + test_param_tag + '.pkl'
samples_per_video = (2 if do_mirror else 1) * (5 if do_crop else 1) * test_segments
assert samples_per_video % test_batch_size == 0
print("test_param_tag is {}".format(test_param_tag))
print("{} data will be written to {}".format(output_layer_name, pickle_file))
print("samples_per_video is {}".format(samples_per_video))

net = caffe.Net(net_prototxt, caffemodel, caffe.TEST)
num_correct = 0

all_output = []
for video_id in range(num_videos):
	video_output = np.zeros([1 * samples_per_video, 101])
	label = -1;
	for offset in range(0, samples_per_video, test_batch_size):	
		net.forward()
		batch_output = net.blobs[output_layer_name].data
		video_output[offset: offset+test_batch_size, :] = batch_output
		if offset == 0:
			label = int(net.blobs['label'].data[0,0,0])
		else:
			assert label == int(net.blobs['label'].data[0,0,0])
	pd_label = np.argmax(video_output.mean(axis=0))
	all_output.append((video_id, video_output, label))
	if pd_label == label:
		num_correct += 1
	print("video_id: {}".format(video_id))
	print("{}".format(np.argmax(video_output, axis=1)))
	print("result: {}, {}".format(pd_label, label))
	print("accuracy: {}/{} {}".format(num_correct, video_id+1, num_correct/float(video_id + 1)))

print("accuracy: {}/{} {}".format(num_correct, num_videos, num_correct/float(num_videos)))

with open(pickle_file, "wb") as f:
	pickle.dump(all_output, f)

print("pkl file is written to {}".format(pickle_file))

