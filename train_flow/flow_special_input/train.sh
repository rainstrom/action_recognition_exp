# train all, init_lr 0.001, step 10000, gamma: 0.1, max_iter: 30000, batch 100
# fc6 *5, fc7,8 *10 lr
# stop at flow_special_iter_17817.caffemodel
# restore fc lr
caffe train -solver ./solver.prototxt -weights ./flow_10img_init.caffemodel -gpu 4 | tee log_train_flow_special_input.txt
