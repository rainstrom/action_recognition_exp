import numpy as np
import sys
import caffe

caffe.set_mode_cpu()
original_net = caffe.Net('./vgg_16_rgb_train_val_fast.prototxt', './caffemodels/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
net = caffe.Net('./conv1_1minus_rgb5/train_val_fast.prototxt', caffe.TEST)
save_to_model = './caffemodels/vgg_16_conv1_1minus_rgb5_init_fromvgg.caffemodel'

transfered = {param: False for param in net.params.keys()}
print(original_net.params.keys())
print(net.params.keys())

for k in original_net.params.keys():
	source = k
	target = k
	if source == "conv1_1":
		target = k + "_p0"
	if source == "conv1_2":
		target = "conv1_2_minus"
	print("{}-->{}".format(source, target))
	
	print("{}{}-->{}{}".format(source, original_net.params[source][0].data.shape, target, net.params[target][0].data.shape))
	print("{}{}-->{}{}".format(source, original_net.params[source][1].data.shape, target, net.params[target][1].data.shape))
	
	if source == "conv1_2":
		for offset in range(0, net.params[target][0].data.shape[1], original_net.params[source][0].data.shape[1]):
			net.params[target][0].data[:, offset: offset + original_net.params[source][0].data.shape[1], ...] = original_net.params[source][0].data / 4
		net.params[target][1].data[...] = original_net.params[source][1].data[...]
	
	else:	
		for i in range(len(net.params[target])):
			net.params[target][i].data[...] = original_net.params[source][i].data[...]
	transfered[target] = True

print("{}".format(transfered))
print("save to {}".format(save_to_model))
net.save(save_to_model)
print("saved")

