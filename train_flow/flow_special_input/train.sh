# train all, init_lr 0.001, step 10000, gamma: 0.1, max_iter: 30000, batch 100
caffe train -solver ./solver.prototxt -weights ./flow_10img_init.caffemodel -gpu 1 | tee log_train_flow_1mean_9minus.txt
