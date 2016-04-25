import numpy as np
import sys
import caffe

caffe.set_mode_cpu()
vggmodel = '../VGG_ILSVRC_16_layers.caffemodel'
vggnet = caffe.Net('./vgg16.prototxt', vggmodel, caffe.TEST)
net = caffe.Net('./train_val_fast.prototxt', caffe.TEST)
save_to_model = './flow1_init.caffemodel'

transfered = {param: False for param in net.params.keys()}
print(vggnet.params.keys())
print(net.params.keys())

for k in vggnet.params.keys():
	source = k
	target = "flow_" + k
	print("vggnet: {}-->{}".format(source, target))
	if source == "fc8_rgb":
		continue
	if source == "conv1_1":
		assert vggnet.params[source][0].data.shape[1] == 3
		assert net.params[target][0].data.shape[1] == 2
		for i in range(2):
			net.params[target][0].data[:,i,...] = np.mean(vggnet.params[source][0].data, axis=1)
			print "conv1_1 from shape:", np.mean(vggnet.params[source][0].data, axis=1).shape
			print "conv1_1 to shape:", net.params[target][0].data[:,i,...].shape
		net.params[target][1].data[...] = vggnet.params[source][1].data[...]
	else:
		for i in range(len(net.params[target])):
			net.params[target][i].data[...] = vggnet.params[source][i].data[...]
	transfered[target] = True

print("{}".format(transfered))
print("save to {}".format(save_to_model))
net.save(save_to_model)
print("saved")

