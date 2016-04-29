# train all, init_lr 0.001, step 10000, gamma: 0.1, max_iter: 30000, batch 100
# fc for high lr 5x for fc6 and 10x for fc7,8
# running
# low dropout rate test, normal fc lr
# runnine
caffe train -solver ./solver.prototxt -weights ./flow_10img_init.caffemodel -gpu 6 | tee log_train_flow_10img.txt

