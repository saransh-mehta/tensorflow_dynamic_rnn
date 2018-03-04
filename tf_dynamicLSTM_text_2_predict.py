# here we will use our saved dynamic LSTM model,
# hence, we will load the model, and use it for prediction
import tensorflow as tf
import os
import numpy as np

SAVE_DIR = os.path.abspath(os.path.join('model_save', 'dynamicLSTM_text_1'))

# at the time of saving, a meta file is created which stores the graph and structure of the original model
# hence, we will use it to reconstruct

# the meta graph can be tell tf about the internal relationsof the variables, but it can't 
# still give the relations with placeholders, hence we will recreate certain placeholders
x = tf.placeholder(shape = [None, TIME_STEPS], dtype = tf.int32, name = 'input')
y = tf.placeholder(shape = [None, CLASS_NUM], dtype = tf.float32, name = 'output')
senLengths = tf.placeholder(shape = [None], dtype = tf.int32, name = 'senLengths')

with tf.Session() as sess:
	loader = tf.train.import_meta_graph(os.path.join(SAVE_DIR, 'model_1-100.meta'))
	# this has loaded the graph, but we still need to load the values of the parameters
	loader.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))
	# tf.train.latest_checkpoint('./') gives out the full path of the latest saved checkpoint
	# now the values of the original network are loaded
	print('successfully loaded')

	# now we will need to regenerate the input and output placeholders
	ans = sess.run(accuracy, feed_dict = {x : ['Four Four Two Four Two'], y : [10]})