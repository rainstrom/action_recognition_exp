import numpy as np
import sys
import caffe
import os
caffe.set_mode_cpu()

print 'Number of arguments:', len(sys.argv), 'arguments.'
assert len(sys.argv) == 2

flow_num = int(sys.argv[1])
assert flow_num > 0
print 'Do net surgery for', flow_num, 'flow images input' 

proj_dir = 'flow_' + str(flow_num) + 'img/'
assert os.path.isdir(proj_dir)

vggnet = caffe.Net('../action-recognition/vgg16.prototxt', '../caffemodels/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)
os.chdir(proj_dir)
net = caffe.Net('train_val_fast.prototxt', caffe.TEST)
os.chdir('../')
# Remember to modify
save_to_model = proj_dir + 'flow_'+str(flow_num)+'img_init.caffemodel'

transfered = {param: False for param in net.params.keys()}
print(vggnet.params.keys())
print(net.params.keys())

for k in vggnet.params.keys():
	source = k
	target = "flow_" + k
	print("vggnet: {}-->{}".format(source, target))
	if source == "fc8":
		# 100 out
		# remember use xaiver or other initialization method
		continue
	if source == "conv1_1":
		# Remember to modify
		conv1_1_feature_map_num = 2 * flow_num
		assert vggnet.params[source][0].data.shape[1] == 3
		assert net.params[target][0].data.shape[1] == conv1_1_feature_map_num
		for i in range(conv1_1_feature_map_num):
			net.params[target][0].data[:,i,...] = np.mean(vggnet.params[source][0].data, axis=1)
			print "conv1_1 from shape:", np.mean(vggnet.params[source][0].data, axis=1).shape
			print "conv1_1 to shape:", net.params[target][0].data[:,i,...].shape
		# net.params[target][1].data[...] = vggnet.params[source][1].data[...]
		net.params[target][1].data[...] = 0
	else:
		for i in range(len(net.params[target])):
			net.params[target][i].data[...] = vggnet.params[source][i].data[...]
	transfered[target] = True

print("{}".format(transfered))
print("save to {}".format(save_to_model))
net.save(save_to_model)
print("saved")

