import numpy as np
import sys
import caffe

caffe.set_mode_cpu()
net1 = caffe.Net('./xl_combine_train_val_fast.prototxt', './thu_action_recognition_vgg_16_split1_simple_combine.caffemodel', caffe.TEST)
net2 = caffe.Net('./xl_combine_train_val_fast.prototxt', './xl_combine_fusion_iter_10000.caffemodel', caffe.TEST)
net = caffe.Net('./xl_combine_train_val_fast.prototxt', caffe.TEST)
save_to_model = './xl_combine_fusion_iter_10000_combine.caffemodel'

transfered = {param: False for param in net.params.keys()}
print(net1.params.keys())
print(net2.params.keys())
print(net.params.keys())

for k in net1.params.keys():
	source = k
	target = k
	if source[:6] == "rgb_fc" or source[:7] == "flow_fc":
		print("net1: {}-->{}".format(source, target))
		for i in range(len(net.params[target])):
			net.params[target][i].data[...] = net1.params[source][i].data[...]
		transfered[target] = True


for k in net2.params.keys():
	source = k
	target = k
	if source[:6] != "rgb_fc" and source[:7] != "flow_fc":
		print("net2: {}-->{}".format(source, target))
		for i in range(len(net.params[target])):
			net.params[target][i].data[...] = net2.params[source][i].data[...]
		transfered[target] = True


print("{}".format(transfered))
print("save to {}".format(save_to_model))
net.save(save_to_model)
print("saved")

