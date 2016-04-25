import numpy as np
import caffe
import sys
import cPickle as pickle

def full_test(caffemodel, net_prototxt, pkl_dir, experiment_name, output_layer_names, device_id, do_mirror, do_crop, test_segments, test_batch_size):
	caffe.set_device(device_id)
	caffe.set_mode_gpu()
	num_videos = 3783
	# num_videos = 20

	# model to be tested
	# caffemodel = './xl_combine_fusion_dropout_finetune_iter_10000.caffemodel'
	# net_prototxt = './xl_combine_dropout_train_val_fast.prototxt'
	# pkl_dir = "./"
	# experiment_name = "xl_combine_dropout_iter_10000"
	# output_layer_names = ["fusion_fc8"]

	# parameters to set
	# do_mirror = False
	# do_crop = False
	# test_segments = 25
	# test_batch_size = 25

	# generate test_param_tag
	test_param_tag = ("mirror" if do_mirror else "no_mirror") + ("_crop" if do_crop else "_no_crop") + "_seg" + str(test_segments)
	pickle_file = 'full_test_' + experiment_name + '_' + test_param_tag + '.pkl'
	samples_per_video = (2 if do_mirror else 1) * (5 if do_crop else 1) * test_segments
	assert samples_per_video % test_batch_size == 0
	print("test_param_tag is {}".format(test_param_tag))
	print("{} data will be written to {}".format(output_layer_names[0], pickle_file))
	print("samples_per_video is {}".format(samples_per_video))

	net = caffe.Net(net_prototxt, caffemodel, caffe.TEST)
	num_corrects = []
	all_outputs = []
	for name in output_layer_names:
		num_corrects.append(0)
		all_outputs.append([])

	for video_id in range(num_videos):
		video_outputs = [np.zeros([1 * samples_per_video, 101]) for name in output_layer_names]
		label = -1;
		for offset in range(0, samples_per_video, test_batch_size):	
			net.forward()
			for i in range(len(output_layer_names)):
				output_layer_name = output_layer_names[i]
				batch_output = net.blobs[output_layer_name].data
				video_outputs[i][offset: offset+test_batch_size, :] = batch_output
			if offset == 0:
				label = int(net.blobs['label'].data[0,0,0])
			else:
				assert label == int(net.blobs['label'].data[0,0,0])

		print("video_id: {}".format(video_id))
		for i in range(len(video_outputs)):
			output_layer_name = output_layer_names[i]
			print("output_layer_name:{}".format(output_layer_name)) 
			pd_label = np.argmax(video_outputs[i].mean(axis=0))
			all_outputs[i].append((video_id, video_outputs[i], label))
			if pd_label == label:
				num_corrects[i] += 1
			print("data: {}".format(np.argmax(video_outputs[i], axis=1)))
			print("result(pd,lb): {}, {}".format(pd_label, label))
			print("current {} accuracy: {}/{} {}".format(output_layer_name, num_corrects[i], video_id+1, num_corrects[i]/float(video_id + 1)))
		print("\n")

	for i in range(len(output_layer_names)):
		print output_layer_names[i]
		print("accuracy: {}/{} {}".format(num_corrects[i], num_videos, num_corrects[i]/float(num_videos)))

	pickle_file = pkl_dir + pickle_file
	with open(pickle_file, "wb") as f:
		for i in range(len(output_layer_names)):
			pickle.dump(all_outputs[i], f)

	print("pkl file is written to {}".format(pickle_file))

