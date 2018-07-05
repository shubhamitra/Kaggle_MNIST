import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

LEARNING_RATE = 0.001
TRAINING_EPOCHS = 3000
BATCH_SIZE = 100
DISPLAY_STEP = 10
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.6
VALIDATION_SIZE = 2000
train=pd.read_csv("C:/Users/titan/Downloads/train (3).csv")

#splitting the data into image and label
images=train.iloc[:,1:].values
images=images.astype(np.float)
#print(images.shape)
labels=train.iloc[:,0].values
labels_count=np.unique(labels).shape[0]
#rint(labels_count)
#converting into one-hot encoding
labels=(np.arange(labels_count) == labels[:,None]).astype(np.float32)
#print(labels[3,:])
#normalize the dataset

images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
#print(image_width)
# Split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
def next_batch(batch_size):    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]
patch_size=5
num_channels=1
depth=64
num_hidden=1024
num_labels=10
X = tf.placeholder('float', shape=[None, image_size])       # mnist data image of shape 28*28=784
Y_gt = tf.placeholder('float', shape=[None, labels_count])    # 0-9 digits recognition => 10 classes

X1 = tf.reshape(X, [-1,image_width , image_height,1])
print(X1.shape)

#weights initialization
layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, 32], stddev=0.1))
layer1_biases = tf.Variable(tf.zeros([32]))
layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, 32, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
layer3_weights = tf.Variable(tf.truncated_normal(
      [image_width // 4 * image_height // 4 * depth, num_hidden], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')
def model(data):
    conv=tf.nn.conv2d(data,layer1_weights,[1,1,1,1],padding='SAME')
    conv=tf.nn.max_pool(value=conv,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
    print(conv.shape)
    print("1")
    conv=tf.nn.relu(conv+layer1_biases)
    conv=tf.nn.dropout(conv, drop_conv)
    conv=tf.nn.conv2d(conv,layer2_weights,[1,1,1,1],padding='SAME')
    conv=tf.nn.max_pool(value=conv,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
  # print(conv.shape)
   ## print("2")
    conv=tf.nn.relu(conv+layer2_biases)
    conv=tf.nn.dropout(conv, drop_conv)
    shape1= conv.get_shape().as_list()
    print(shape1)
    hidden= tf.reshape(conv, [-1, 3136])
    
    hidden = tf.nn.relu(tf.matmul(hidden, layer3_weights) + layer3_biases)
    hidden=tf.nn.dropout(hidden, drop_conv)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
logits = model(X1)
loss = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(labels=Y_gt, logits=logits))
#regularizer = (tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases))
#loss += 5e-4 * regularizer
#optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

optimizer=tf.train.RMSPropOptimizer(0.001).minimize(loss)  
  # Predictions for the training, validation, and test data.
prediction = tf.nn.softmax(logits)
correct_predict = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
predict = tf.argmax(prediction, 1)

num_steps = 1001   
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []
for i in range(TRAINING_EPOCHS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%DISPLAY_STEP == 0 or (i+1) == TRAINING_EPOCHS:
        
        train_accuracy = accuracy.eval(feed_dict={X:batch_xs, 
                                                  Y_gt: batch_ys,
                                                  drop_conv: DROPOUT_CONV, 
                                                  drop_hidden: DROPOUT_HIDDEN})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ X: validation_images[0:BATCH_SIZE], 
                                                            Y_gt: validation_labels[0:BATCH_SIZE],
                                                            drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        
        # increase DISPLAY_STEP
        if i%(DISPLAY_STEP*10) == 0 and i:
            DISPLAY_STEP *= 10
    # train on batch
    sess.run(optimizer, feed_dict={X: batch_xs, Y_gt: batch_ys, drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})


# check final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={X: validation_images, 
                                                   Y_gt: validation_labels,
                                                   drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
    print('validation_accuracy => %.4f'%validation_accuracy)
test=pd.read_csv("C:/Users/titan/Downloads/test (1).csv")
test_images = test.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={X: test_images, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={X: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], drop_conv: 1.0, drop_hidden: 1.0})


# save results
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

sess.close()
#validation_ccuracy=0.9865 Rmsprop test_accuracy=0.98685
