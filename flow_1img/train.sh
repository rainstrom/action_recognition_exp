#highdrop exp
caffe train -solver ./solver.prototxt -weights ./flow1_init.caffemodel -gpu 1 | tee log_train_flow1.txt
