import numpy as np
import sys
import caffe

caffe.set_mode_cpu()
flow_net = caffe.Net('../../action-recognition/vgg_16_flow_train_val_fast.prototxt', '../../caffemodels/cuhk_action_recognition_16_split1_flow_iter_30000_acc_71.caffemodel', caffe.TEST)
net = caffe.Net('./train_val_fast.prototxt', caffe.TEST)
save_to_model = './flow5_minus_atconv1_1_init.caffemodel'

transfered = {param: False for param in net.params.keys()}
print(flow_net.params.keys())
print(net.params.keys())

for k in flow_net.params.keys():
	source = k
	target = 'flow_' + k
	if source == 'conv1_1_flow':
		target = 'flow_conv1_1'
	if source == 'fc8_flow':
		target = 'flow_fc8'
	print("{}-->{}".format(source, target))
	for i in range(len(net.params[target])):
		net.params[target][i].data[...] = flow_net.params[source][i].data[...]
	transfered[target] = True


print("{}".format(transfered))
print("save to {}".format(save_to_model))
net.save(save_to_model)
print("saved")

