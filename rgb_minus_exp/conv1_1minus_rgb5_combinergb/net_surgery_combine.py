import numpy as np
import sys
import caffe

caffe.set_mode_cpu()
rgb_net = caffe.Net('./vgg_16_rgb_train_val_fast.prototxt', './caffemodels/cuhk_action_recognition_vgg_16_split1_rgb_iter_10000_acc_77.caffemodel', caffe.TEST)
rgb5_net = caffe.Net('./conv1_1minus_rgb5/train_val_fast.prototxt', './conv1_1minus_rgb5/conv1_1minus_rgb5_initfromvgg_split1_iter_25000.caffemodel', caffe.TEST)
net = caffe.Net('./conv1_1minus_rgb5_combinergb/train_val_fast.prototxt', caffe.TEST)
save_to_model = './caffemodels/vgg_16_conv1_1minus_rgb5_combine_rgb_init.caffemodel'

transfered = {param: False for param in net.params.keys()}
print(rgb_net.params.keys())
print(rgb5_net.params.keys())
print(net.params.keys())

for k in rgb_net.params.keys():
	source = k
	target = "rgb_" + k
	if source == "fc8_rgb":
		continue
	print("{}-->{}".format(source, target))	
	
	print("{}{}-->{}{}".format(source, rgb_net.params[source][0].data.shape, target, net.params[target][0].data.shape))
	print("{}{}-->{}{}".format(source, rgb_net.params[source][1].data.shape, target, net.params[target][1].data.shape))
	
	for i in range(len(net.params[target])):
		net.params[target][i].data[...] = rgb_net.params[source][i].data[...]
	transfered[target] = True

for k in rgb5_net.params.keys():
	source = k
	target = "rgb5_" + k
	if source == "fc8_rgb":
		continue
	print("{}-->{}".format(source, target))
	
	print("{}{}-->{}{}".format(source, rgb5_net.params[source][0].data.shape, target, net.params[target][0].data.shape))
	print("{}{}-->{}{}".format(source, rgb5_net.params[source][1].data.shape, target, net.params[target][1].data.shape))
	
	for i in range(len(net.params[target])):
		net.params[target][i].data[...] = rgb5_net.params[source][i].data[...]
	transfered[target] = True

print("{}".format(transfered))
print("save to {}".format(save_to_model))
net.save(save_to_model)
print("saved")

