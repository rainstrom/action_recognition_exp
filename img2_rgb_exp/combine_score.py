fpkl1 = "./full_test_rgb_original_no_mirror_no_crop_seg25.pkl"
fpkl2 = "./conv1_1minus_rgb5/full_test_conv1_1minus_rgb5_initfromvgg_iter30000_no_mirror_no_crop_seg25.pkl"

import cPickle as pickle
import numpy as np

with open(fpkl1, "rb") as f:
	pkl1 = pickle.load(f)

with open(fpkl2, "rb") as f:
	pkl2 = pickle.load(f)

total = 0
kind_total = [0] * 101
correct = 0
kind_correct = [0] * 101
correct1 = 0
kind_correct1 = [0] * 101
correct2 = 0
kind_correct2 = [0] * 101

def softmax(m):
    #print len(m.shape), m.shape
    assert len(m.shape) == 2 and m.shape[1] == 101
    for i in range(m.shape[0]):
        m[i,:] = np.exp(m[i,:])
        m[i,:] = m[i,:]/sum(m[i,:])
    return m
#ATTENTION softmax
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

for i in range(len(pkl1)):
	assert pkl1[i][2] == pkl2[i][2]
	label = pkl1[i][2]
	score1 = np.mean(pkl1[i][1], axis = 0)
	score2 = np.mean(pkl2[i][1], axis = 0)
	score = score1 + score2
	pd_label = np.argmax(score)
	pd_label1 = np.argmax(score1)
	pd_label2 = np.argmax(score2)
	if label == pd_label:
		correct += 1
		kind_correct[label] += 1
	if label == pd_label1:
		correct1 += 1
		kind_correct1[label] += 1
	if label == pd_label2:
		correct2 += 1
		kind_correct2[label] += 1
	total +=1
	kind_total[label] += 1

for i in range(101):
	print("{}: \t{} \t{} \t{}".format(i, kind_correct[i]/float(kind_total[i]),kind_correct1[i]/float(kind_total[i]),kind_correct2[i]/float(kind_total[i])))

#for i in range(101):
#	print("kind {}".format(i))
#	for j in range(len(pkl1)):
#		if pkl1[j][2] == i:
#			print("{}: kind {} example".format(j,i))
#			print np.mean(pkl1[j][1], axis = 0)
#			print np.mean(pkl2[j][1], axis = 0)

print("weighted sum:")
print("{}/{} {}".format(correct, total, correct/float(total)))
print("{}".format(fpkl1))
print("{}/{} {}".format(correct1, total, correct1/float(total)))
print("{}".format(fpkl2))
print("{}/{} {}".format(correct2, total, correct2/float(total)))
