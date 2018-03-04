# here we create dummy text data to use RNN on classifying it. 
# our artificial data has 2 classes, odd and even. odd class has sentences composed of odd digits in words
# and even class has sentences of even digits in words.
import tensorflow as tf
import numpy as np
import os

# hyper parameters
CLASS_NUM = 2
ELEMENT_SIZE = 1 # each word is considered
EMBEDDING_DIMS = 64  
HIDDEN_UNITS = 32
TIME_STEPS = 6  # we are considereding our context size to be 6 words
BATCH_SIZE = 128 
LEARNING_RATE = 0.001 # for RMSPropOptimizer
EPOCHS = 1000
SEED = 2  # for random fn
np.random.seed(SEED)

LOG_DIR = os.path.abspath('tmp/log_dynamicLSTM_text_1')
SAVE_DIR = os.path.abspath(os.path.join('model_save', 'dynamicLSTM_text_1'))
# here randomly we ll create sentences of variable lengths
# so first we ll pad them to equal length, later we will use dynamic rnn to remove the redundancy created
# by padding
digitWordMap = {0:'PAD', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight',9:'nine'}

class Data:
	def __init__(self, senNum):

		self.senNum = senNum
		self.evenSenList = []
		self.oddSenList = []
		self.allData = []
		self.labels = []
		#self.labelsOneHot = np.array()
		self.senLenList = []
		self.wordIndexDicti = {}
		self.indexWordDicti = {}
		self.vocabSize = 0

	def create_numbers(self):

		for i in range(self.senNum):
			randomLength = np.random.randint(3, 7)
			# we ll make length vary randomly from 3 to 6
			self.senLenList.append(randomLength)

			randomOddNumbers = np.random.choice(range(1, 10, 2), randomLength)
			randomEvenNumbers = np.random.choice(range(2, 10, 2), randomLength)

			#padding
			if randomLength < 6:

				pad = np.zeros(6 - randomLength)
				#print(randomLength, np.shape(pad))
				randomOddNumbers1 = np.hstack((randomOddNumbers, pad))
				randomEvenNumbers1 = np.hstack((randomEvenNumbers, pad))

			self.evenSenList.append(" ".join([digitWordMap[i] for i in randomEvenNumbers1]))
			self.oddSenList.append(" ".join([digitWordMap[i] for i in randomOddNumbers1]))

		self.allData = self.evenSenList + self.oddSenList
		self.labels = [1]*10000 + [0]*10000
		self.senLenList = self.senLenList + self.senLenList
		# 1 for even, 0 for odd
		#return self.allData, self.labels, self.senLenList
	def create_vocab(self):
		# here we will map each unique word present in data to a unique integer
		# this is a common conventional technique of NLP, later we will use vectors for
		# more precision
		index = 0
		for sentence in self.allData:
			#print (type(sentence))
			for word in sentence.lower().split():

				if word not in self.wordIndexDicti:
					self.wordIndexDicti[word] = index
					self.indexWordDicti[index] = word
					index += 1

		self.vocabSize = len(self.wordIndexDicti)
		#return self.wordIndexDicti

	def one_hot_encoder(self, labelsList, classes = 2):
		n = len(labelsList)
		out = np.zeros((n, classes))
		out[range(n), labelsList] = 1
		self.labelsOneHot = out
		#return self.labelsOneHot # returns one hot in numpy array form

	def train_test_split(self, percentTrain):
		from sklearn.utils import shuffle
		self.allData,self.labelsOneHot,self.senLenList=shuffle(self.allData,self.labelsOneHot,self.senLenList,random_state=SEED)

		trainX = self.allData[:int(len(self.allData)*(percentTrain))]
		trainY = self.labelsOneHot[:int(len(self.allData)*(percentTrain))]
		trainSenLenList = self.senLenList[:int(len(self.allData)*(percentTrain))]

		testX = self.allData[int(len(self.allData)*(percentTrain)) :]
		testY = self.labelsOneHot[int(len(self.allData)*(percentTrain)) :]
		testSenLenList = self.senLenList[int(len(self.allData)*(percentTrain)) :]

		return trainX, trainY,trainSenLenList, testX, testY, testSenLenList

	def get_next_batch(self, batchSize, dataX, dataY, dataSenLenList):
		# this fn will take out batches of batchSize from training data
		indexes = list(range(len(dataX)))
		np.random.shuffle(indexes)
		batch = indexes[:batchSize]
		# now the trick is to convert the words into their respective integer through
		# wordIndexMap and then feed into Rnn
		X = [ [self.wordIndexDicti[word] for word in dataX[i].lower().split() ] for i in batch]
		Y = [ dataY[i] for i in batch]
		seqLens = [ dataSenLenList[i] for i in batch]
		return X, Y, seqLens

data = Data(10000)
data.create_numbers()
#print(data.allData)
data.create_vocab()
data.one_hot_encoder(data.labels)
trainX, trainY,trainSenLenList, testX, testY, testSenLenList = data.train_test_split(0.5)

# creating tensorflow dynamic RNN model

#placeholders
with tf.name_scope('placeholders') as scope:
	x = tf.placeholder(shape = [None, TIME_STEPS], dtype = tf.int32, name = 'input')
	y = tf.placeholder(shape = [None, CLASS_NUM], dtype = tf.float32, name = 'output')
	# here we need one more placeholder which will keep record of sentence length in each given sentence
	# so tht dynamic rnn can work according to it
	senLengths = tf.placeholder(shape = [None], dtype = tf.int32, name = 'senLengths')

with tf.name_scope('embeddings') as scope:
	# embeddings can be thought of as a lookup table, mapping words to their dense vector values
	# a mapping frm high dimensional one hot vector encoding to lower dimension dense vectors
	# optimized these vectors are a part of training
	embeddings = tf.Variable(tf.random_uniform([data.vocabSize, EMBEDDING_DIMS], -1.0, 1.0,
		name = 'embedding'))
	embed = tf.nn.embedding_lookup(embeddings, x)

with tf.name_scope('LSTM') as scope:
	cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_UNITS, forget_bias = 1.0)
	# or we could use cell = tf.contrib.BasicLSTMCell(hidden_size)
	initialState = cell.zero_state(BATCH_SIZE, dtype = tf.float32)
	outputTensor, lastState = tf.nn.dynamic_rnn(cell, embed, sequence_length = senLengths,
												 initial_state = initialState, dtype = tf.float32)

with tf.name_scope('linear_layer') as scope:
	# here instead of making manually weights and biases, we ll be using tf.layers.dense()
	# BUT REMEMBER THT ANYTHING WE DECLARE AS VARIABLE IN TF, GETS TRAINED, I.E UPDATED DURING TRAINING
	finalOutput = tf.layers.dense(lastState[1], units = CLASS_NUM, activation = None)

with tf.name_scope('train') as scope:
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = finalOutput, labels = y))
	optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)
	train = optimizer.minimize(loss)

with tf.name_scope('accuracy') as accuracy:
	correctPrediction = tf.equal(tf.argmax(finalOutput, axis = 1), tf.argmax(y, axis = 1))
	accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32)) * 100

 
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

	sess.run(init)
	trainWriter = tf.summary.FileWriter(LOG_DIR + '/train', sess.graph)
	testWriter = tf.summary.FileWriter(LOG_DIR + '/test', sess.graph)

	# here we will also save some important variable in a collection that will be required
	# when we will load saved model to take out accuracy, all placeholders related to accuracy 
	# will be required

	#tf.add_to_collection('requiredVariables', x)
	#tf.add_to_collection('requiredVariables', y)
	#tf.add_to_collection('requiredVariables', senLengths)

	for i in range(EPOCHS):

		batchX, batchY, seqLens = data.get_next_batch(BATCH_SIZE, trainX, trainY, trainSenLenList)
		sess.run(train, feed_dict = {x : batchX, y : batchY, senLengths : seqLens})

		saver.save(sess, os.path.join(SAVE_DIR, 'model_1'), global_step = 100)


		if i % 100 == 0:
			# calculating train accuracy
			acc, lossTmp = sess.run([accuracy, loss], feed_dict = {x : batchX, y : batchY, senLengths : seqLens})
			print('Iter: '+str(i)+' Minibatch_Loss: '+"{:.6f}".format(lossTmp)+' Train_acc: '+"{:.5f}".format(acc))

	for i in range(5):
		# calculating test accuracy
		testBatchX, testBatchY, testSeqLens = data.get_next_batch(BATCH_SIZE, testX, testY, testSenLenList)
		testAccuracy = sess.run(accuracy, feed_dict = {x : testBatchX, y : testBatchY, senLengths: testSeqLens})
		print('test accuracy : ', testAccuracy)
