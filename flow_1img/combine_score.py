tags = ["rgb_original", "flow_original", "flow1"]
fpkls= ["../../full_test_rgb_original_no_mirror_no_crop_seg25.pkl",\
		"../../full_test_flow_original_no_mirror_no_crop_seg25.pkl",\
		"./flow1_no_mirror_no_crop_seg25.pkl"]
score_weights = [[1, 1, 0], [1, 2, 0], [1, 0, 1], [1, 0, 2]]
source_num = len(tags)
exp_num = len(score_weights)

import cPickle as pickle
import numpy as np

assert source_num == len(fpkls)


pkls = []
for fpkl in fpkls:
	print("reading {}".format(fpkl))
	with open(fpkl, "rb") as f:
		pkls.append(pickle.load(f))

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

for exp_id in range(exp_num):
	total = 0
	correct = 0
	corrects = [0] * source_num

	# kind_total = [0] * 101
	# kind_correct = [0] * 101
	# kind_corrects = {tag: [0] * 101 for tag in tags}

	for i in range(len(pkls[0])):
		label = pkls[0][i][2]
		for source_id in range(1, source_num):
			assert pkls[source_id][i][2] == label
		
		scores = [np.mean(pkls[source_id][i][1], axis = 0) * score_weights[exp_id][source_id] for source_id in range(source_num)]
		
		for source_id in range(source_num):
			pd_label = np.argmax(scores[source_id])
			if pd_label == label:
				# kind_corrects[tags[j]][label] += 1
				corrects[source_id] += 1

		sum_score = np.sum(scores, axis=0)
		pd_label = np.argmax(sum_score)
		if pd_label == label:
			# kind_correct[label] += 1
			correct += 1
		total +=1
		# kind_total[label] += 1

	print("exp {}: {},{}".format(exp_id, tags, score_weights[exp_id]))
	print("weighted sum: {}/{} {}".format(correct, total, correct/float(total)))
	for source_id in range(source_num):
		print("{}: {}/{} {}".format(tags[source_id], corrects[source_id], total, corrects[source_id]/float(total)))
	print("\n")
