import scipy.io
import numpy as np
from sklearn.cluster import KMeans
from sklearn import svm
import time
import os
import subprocess

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

# print "sidush"

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
		print (l)
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
		prediction = clf.predict([datapoint])[0]
		print ("Prediction:",prediction)
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
		print (l)
		temp = [0,0]
		H_test.extend([temp]*l)
	H_test = np.transpose(np.array(H_test))
	testing_feats = np.transpose(np.array(testing_feats))



train_data = build_dict(mat['TR1'])
test_data = build_dict(mat['TS1'])


construct_node_data(node_data, headnode, 14)
# construct_svm_all_nodes(node_data)

# accuracy = get_accuracy(test_data)

# print ("Accuracy:",accuracy)


# Build test data

H_test = []
testing_feats = []
for key in test_data.keys():
	testing_feats.extend(test_data[key])
	l = test_data[key].shape[0]
	print (l)
	if key == 0 or key == 1 or key == 6:
		temp = [0,1]
	else:
		temp = [1,0]
	H_test.extend([temp]*l)
H_test = np.transpose(np.array(H_test))
testing_feats = np.transpose(np.array(testing_feats))


def matlab_vaali_backchodi(id,sleep_time = 264):
	for key_tuple in node_data.keys():
		print ("Starting Tuple:",str(key_tuple))
		data = node_data[key_tuple]
		H_train = data["H_train"]
		training_feats = data["training_feats"]
		
		dest = "C:\\Users\\biplab\\Desktop\\Group7_BTP\\sharingcode-LCKSVD\\trainingdata\\features_apne.mat"
		scipy.io.savemat(dest, mdict={'H_train': H_train, 'training_feats': training_feats, 'H_test': H_test, 'testing_feats': testing_feats})

		# Call Matlab Here
		#######################################

		# p = subprocess.Popen("matlab -nodisplay -nosplash -nodesktop -r \"run('C:\\Users\\biplab\\Desktop\\Group7_BTP\\sharingcode-LCKSVD\\main.m apne')\"",stdout=subprocess.PIPE, shell=True)
		# p = subprocess.Popen("matlab -nodisplay -nosplash -nodesktop -r \"run('main.m')\" apne",stdout=subprocess.PIPE, shell=True)

		p = subprocess.Popen('matlab -nodisplay -nosplash -nodesktop -r "main '+id+'"')
		time.sleep(sleep_time)
		
		# Save the contents
		temp = scipy.io.loadmat("./trainingdata/prediction.mat")
		filename = "prediction_"+str(key_tuple)[1:-1]+".mat"
		scipy.io.savemat(filename,temp)
		os.system("taskkill /im matlab.exe")
		break

matlab_vaali_backchodi("apne")











