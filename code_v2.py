import scipy.io
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
import time
import subprocess
import os
from threading import Thread
import pickle
import pdb

mat = scipy.io.loadmat('./data_hyper.mat')
node_data = {}
headnode = tuple(range(14))



def build_dict(arr):
	temp = {}
	for i in range(len(arr)):
		temp[i] = arr[i][0]
	return temp

def get_means(arr, lt=False):
	means = {}
	for key in arr.keys():
		means[key] = np.mean(arr[key], axis=0)
	return means


def get_means_list(train_data, current_node):
	means = []
	for i in current_node:
		means.append(np.mean(train_data[i], axis=0))
	return means

def construct_node_data(node_data, current_node, N):
	
	print (node_data.keys())

	if len(current_node) <= 1:
		return

	node_data[current_node] = {}

	# Build Training Data
	means_list = get_means_list(train_data, current_node)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(means_list)
	means_classes = kmeans.predict(means_list)
	training_feats = []
	H_train = []

	for i,key in enumerate(current_node):
		training_feats.extend(train_data[key])
		l = train_data[key].shape[0]
		# print (l)
		temp = [0,0]
		temp[means_classes[i]] = 1
		H_train.extend([temp]*l)

	left_classes = []
	right_classes = []

	for i in range(len(current_node)):
		if means_classes[i] == 0:
			left_classes.append(current_node[i])
		else:
			right_classes.append(current_node[i])

	node_data[current_node]["H_train"] = np.transpose(np.array(H_train))
	node_data[current_node]["training_feats"] = np.transpose(np.array(training_feats))
	node_data[current_node][0] = tuple(left_classes)
	node_data[current_node][1] = tuple(right_classes)

	construct_node_data(node_data, node_data[current_node][0], N)
	construct_node_data(node_data, node_data[current_node][1], N)

def construct_svm_all_nodes(node_data, kernel="poly"):
	for current in node_data.keys():
		data = node_data[current]
		train = np.transpose(data["H_train"])
		training_feats = np.transpose(data["training_feats"])
		H_train = []
		for temp in train:
			if temp[0] == 1:
				H_train.append(0)
			else:
				H_train.append(1)
		H_train = np.array(H_train)
		clf = svm.SVC(kernel = kernel)
		clf.fit(training_feats, H_train)
		node_data[current]["classifier"] = clf

def classify_datapoint(datapoint, node_data = node_data, headnode = headnode):
	current_node = headnode

	while len(current_node) > 1:
		clf = node_data[current_node]["classifier"]
		prediction = clf.predict(datapoint)[0]
		# print ("Prediction:",prediction)
		current_node = node_data[current_node][prediction]
	return current_node[0]

def predict(datapoints):
	result = []
	for datapoint in datapoints:
		result.append(classify_datapoint(datapoint))
	return result

def get_accuracy(test_data):
	total = 0
	correct = 0
	for key in test_data.keys():
		class_data = test_data[key]
		for datapoint in class_data:
			total += 1
			if classify_datapoint(datapoint) == key:
				correct += 1
	accuracy = float(correct)/float(total)
	print ("****"*20,"\n",correct,"samples correctly classified out of total",total,"samples","\n","****"*20)
	return accuracy

def construct_test_data(test_data):
	H_test = []
	testing_feats = []
	for key in test_data.keys():
		testing_feats.extend(test_data[key])
		l = test_data[key].shape[0]
		# print (l)
		temp = [0,0]
		H_test.extend([temp]*l)
	H_test = np.transpose(np.array(H_test))
	testing_feats = np.transpose(np.array(testing_feats))

def start_matlab(id_temp, sparsitythres, sqrt_alpha, sqrt_beta, iterations4ini):
	p = subprocess.Popen('matlab -nodisplay -nosplash -nodesktop -r \"main '+id_temp+' '+sparsitythres+' '+sqrt_alpha+' '+sqrt_beta+' '+iterations4ini+'\"')


def matlab_vaali_backchodi(sleep_time = 150, sparsitythres="26", sqrt_alpha="5", sqrt_beta="3", iterations4ini="20", distinction_id=""):
	
	for key_tuple in node_data.keys():
		print ("Starting:",str(key_tuple))
		data = node_data[key_tuple]
		H_train = data["H_train"]
		training_feats = data["training_feats"]
		id_temp = str(key_tuple)[1:-1]
		id_temp  = id_temp.replace(', ','_')
		id_temp = id_temp+distinction_id
		# print(id_temp)
		dest = "./trainingdata/features_"+id_temp+".mat"
		scipy.io.savemat(dest, mdict={'H_train': H_train, 'training_feats': training_feats, 'H_test': H_test, 'testing_feats': testing_feats})

		# Call Matlab Here
		thread = Thread(target = start_matlab, args = (id_temp, sparsitythres, sqrt_alpha, sqrt_beta, iterations4ini))
		thread.start()

	time.sleep(sleep_time)
	# Save the contents
	for key_tuple in node_data.keys():
		id_temp = str(key_tuple)[1:-1]
		id_temp  = id_temp.replace(', ','_')
		id_temp = id_temp+distinction_id
		temp = scipy.io.loadmat("./trainingdata/prediction_"+id_temp+".mat")
		filename = "prediction_"+id_temp+".mat"
		scipy.io.savemat(filename,temp)
	os.system("taskkill /im matlab.exe")
	time.sleep(25)
	

def set_prediction_data_lcksvd(distinction_id=""):
	for key_tuple in node_data.keys():
		id_temp = str(key_tuple)[1:-1]
		id_temp  = id_temp.replace(', ','_')
		id_temp = id_temp+distinction_id
		filename = "prediction_"+id_temp+".mat"
		prediction = ((scipy.io.loadmat(filename)['prediction2']) - 1)
		node_data[key_tuple]["prediction"] = prediction

def get_prediction_lcksvd(testing_feats, node_data, headnode):
	testing_data = np.transpose(testing_feats)
	point_index = 0
	correct = 0
	for key in test_data.keys():
		for point in test_data[key]:
			current_node = headnode
			while len(current_node) > 1:
				# print (node_data[current_node]["prediction"],point_index)
				pred = node_data[current_node]["prediction"][0][point_index]
				# print ("Prediction:",pred)
				current_node = node_data[current_node][pred]
			if current_node[0] == key:
				correct += 1
			point_index += 1
	accuracy = float(correct)/float(point_index)
	print ("****"*20,"\n",correct,"samples correctly classified out of total",point_index,"samples","\n","****"*20)
	print ("Accuracy:",accuracy)
	return accuracy


train_data = build_dict(mat['TR1'])
test_data = build_dict(mat['TS1'])


construct_node_data(node_data, headnode, 14)

test_svm = False

if test_svm:
	construct_svm_all_nodes(node_data)

	accuracy = get_accuracy(test_data)

	print ("SVM Accuracy:",accuracy)



# Build test data

H_test = []
testing_feats = []
for key in test_data.keys():
	testing_feats.extend(test_data[key])
	l = test_data[key].shape[0]
	# print (l)
	temp = [0,0]
	H_test.extend([temp]*l)
H_test = np.transpose(np.array(H_test))
testing_feats = np.transpose(np.array(testing_feats))


def start_process(results, sleep_time = 150, sparsitythres="26", sqrt_alpha="5", sqrt_beta="3", iterations4ini="20", distinction_id=""):
	# matlab_vaali_backchodi(sleep_time=sleep_time, sparsitythres=sparsitythres, sqrt_alpha=sqrt_alpha, sqrt_beta=sqrt_beta, iterations4ini=iterations4ini, distinction_id=str(distinction_id))
	set_prediction_data_lcksvd(distinction_id = distinction_id)
	accuracy = get_prediction_lcksvd(testing_feats, node_data, headnode)
	results[tuple((sparsitythres, sqrt_alpha, sqrt_beta, iterations4ini))] = accuracy
	print (tuple((sparsitythres, sqrt_alpha, sqrt_beta, iterations4ini)),":",accuracy)

def optimize_parameters(results):
	start = time.time()
	sparsitythres=[18, 26, 30]
	sqrt_alpha=[2, 5, 8, 10, 12, 18]
	sqrt_beta=[3, 5, 7, 9, 11, 18]
	iterations4ini=[30]
	# sparsitythres=[18, 26]
	# sqrt_alpha=[2, 5]
	# sqrt_beta=[5]
	# iterations4ini=[20]

	count = 0
	for p in sparsitythres:
		for q in sqrt_alpha:
			for r in sqrt_beta:
				for s in iterations4ini:
					print ((p,q,r,s))
					distinction_id = "__"+str(count)
					# thread = Thread(target = start_process, args = (results, 100, str(p), str(q), str(r), str(s), str(distinction_id)))
					# thread.start()
					start_process(results, sleep_time = 90, sparsitythres=str(p), sqrt_alpha=str(q), sqrt_beta=str(r), iterations4ini=str(s), distinction_id=str(distinction_id))
					count +=1
	end = time.time()
	print ("DONE!")
	print ("Time Taken:",end-start)
	print (results)
	pickle.dump(results, open("dumped_var","wb"))

results = {}
optimize_parameters(results)

pdb.set_trace()
