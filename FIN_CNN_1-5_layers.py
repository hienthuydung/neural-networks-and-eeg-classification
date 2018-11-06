
# coding: utf-8

# In[ ]:


#import modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.stats import zscore
import sklearn
from sklearn.model_selection import train_test_split
import fnmatch


#functions for finding and reading data files

def find_files(directory, pattern='Data*.csv', withlabel=True):
    '''fine all the files in one directory and assign '1'/'0' to F or N files'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            if withlabel:
                if 'non-seizure' in filename:
                    label = '0'
                else:
                    label = '1'
                files.append((os.path.join(root, filename), label))
            else:  # only get names
                files.append(os.path.join(root, filename))
    
    return files

def read_data(filename, header=None, ifnorm=False):
    '''read data from .csv
    Param:
        filename: string e.g. 'data/Data_F_Ind0001.csv'
        ifnorm: boolean, 1 zscore normalization
        start: with filter augmented data, start index of the column indicate which group of data to use
        width: how many columns of data to use, times of 2
    return:
        data: 2d array [seq_len, channel]'''

    data = pd.read_csv(filename, header=header, nrows=None)
    data = data.values  ### get data without row_index
    if ifnorm:   ### 2 * 10240  normalize the data into [0.0, 1.0]]
        data_norm = zscore(data)
        data = data_norm
    #data = np.squeeze(data)   ## shape from [1, seq_len] --> [seq_len,]
    return data


#find segmented files and return them with label
seg_files_wlabel = find_files('C:\\Users\\hien\\Desktop\\eeg_data\\seg_data', pattern='seg*.csv', withlabel=True)


#split files and labels
files, labels = np.array(seg_files_wlabel)[:, 0].astype(np.str), np.array(np.array(seg_files_wlabel)[:, 1]).astype(np.int)


#split data into training and testing
files_train, files_test, labels_train, labels_test = train_test_split(files, labels, test_size=0.4, random_state=42)



#set parameters

learning_rate = 0.001
total_epochs = 30 #number of iterations over all samples / epoch
batch_size = 64 #amount of files fed to model per iteration
test_sample = len(labels_test)
total_batches = len(labels_train)//batch_size
num_classes = 2


#set variables x=input, y_=output, W=weights, b=bias

# Placeholder variable for the input images
x = tf.placeholder("float",[None, 2048, 1], name = "x") #number of data points
# Reshape it into [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x, [-1, 2048, 1, 1]) #why reshape?

# Placeholder variable for the true labels associated with the images
y_true = tf.placeholder("float",[None,2], name = "y_true") #2 possible outputs
y_true_cls = tf.argmax(y_true, axis=1) 



#set var

filter_no = 4
filtersize = 9
poolsize = 4
stride = 1
poolstride = 2



#set up network


layer = tf.layers.conv2d(x_image, filter_no, [filtersize, 1], strides = (stride, 1), padding = "same", activation = tf.nn.relu)

layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[poolsize, 1], strides= [poolstride, 1])

layer = tf.layers.conv2d(layer, filter_no, [filtersize, 1], strides = (stride, 1), padding = "same", activation = tf.nn.relu)

layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[poolsize, 1], strides= [poolstride, 1])

layer = tf.layers.conv2d(layer, filter_no, [filtersize, 1], strides = (stride, 1), padding = "same", activation = tf.nn.relu)

layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[poolsize, 1], strides= [poolstride, 1])#poolsize/strides: 2, 4, 6... filter size

layer = tf.layers.conv2d(layer, filter_no, [filtersize, 1], strides = (stride, 1), padding = "same", activation = tf.nn.relu)

layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[poolsize, 1], strides= [poolstride, 1])

layer = tf.layers.conv2d(layer, filter_no, [filtersize, 1], strides = (stride, 1), padding = "same", activation = tf.nn.relu)

layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[poolsize, 1], strides= [poolstride, 1])

flatten_layer = tf.layers.flatten(layer, name=None)

layer = tf.layers.dense(flatten_layer, 100, activation=tf.nn.relu)

layer = tf.nn.dropout(layer, keep_prob = 0.3, noise_shape=None, seed=None, name=None)

output = tf.layers.dense(layer, 2, activation=tf.nn.softmax)


network_output = tf.argmax(output, axis=1) 



#define and minimze cost function

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_true)
loss_op = tf.reduce_mean(cross_entropy)



#training

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op) 



#Determining accuracy of parameters

correct_prediction = tf.equal(tf.argmax(output, axis=1), y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#rearrange dataset

dataset_train = tf.data.Dataset.from_tensor_slices((files_train, labels_train)).repeat().batch(batch_size).shuffle(buffer_size=10000)
iter = dataset_train.make_initializable_iterator()
ele = iter.get_next()   #you get the filename



#run session and plot train accuracy

with tf.Session() as sess:
    sess.run(iter.initializer)
    sess.run(tf.global_variables_initializer())
    
    acc_epoch = np.zeros((total_epochs)) 
    acc_epoch_test = np.zeros((total_epochs)) 
    
    for epoch in range(total_epochs):
        acc_batch = 0
        acc_batch_test = 0
        avg_cost = 0
    
        for batch in range(total_batches):
            files_train, labels_train =  sess.run(ele)   # names, 1s/0s the filename is bytes object!!! TODO
            
            data_train = np.zeros([batch_size, 2048, 1])
            files_train = files_train.astype(np.str)
            
            for ind in range(batch_size):
                
                data = read_data(files_train[ind],  header=None, ifnorm=True)
                data_train[ind, :] = data
            
            labels_train_hot =  np.eye((num_classes))[labels_train.astype(int)] # get one-hot lable
            
            acc, _, pred, loss = sess.run([accuracy, optimizer, correct_prediction, loss_op], feed_dict={x: data_train, y_true: labels_train_hot})
        
        
            acc_batch += acc 
    
        acc_epoch[epoch] = acc_batch / total_batches
        avg_cost += loss / total_batches
        if epoch % 10 == 0:
            print("acc train", acc_batch / total_batches)
        
        data_test = np.zeros([test_sample, 2048, 1])
        for test_nr in range(test_sample):
            
            files_test = files_test.astype(np.str)
                
            data_test_01 = read_data(files_test[test_nr],  header=None, ifnorm=True)
            data_test[test_nr, :] = data_test_01
            
        labels_test_hot = np.eye((num_classes))[labels_test.astype(int)]
        
        acc_test, pred_test, pred_labels = sess.run([accuracy, correct_prediction, network_output], feed_dict={x: data_test, y_true: labels_test_hot})
        
        #acc_batch_test += acc_test 
    
        acc_epoch_test[epoch] = acc_test 
    print("acc test", acc_epoch_test)
        
        
        
            

            
plt.plot(np.array(acc_epoch))   
plt.plot(np.array(acc_epoch_test))  


#confusion marix
sklearn.metrics.confusion_matrix(labels_test, pred_labels, labels=None, sample_weight=None)

