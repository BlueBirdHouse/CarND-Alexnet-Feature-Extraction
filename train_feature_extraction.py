import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import time
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt


#%%函数定义区
def evaluate(X_data, y_data,sess,logits,one_hot_y,BATCH_SIZE):
    '''
    logits:网络的直接输出
    one_hot_y：验证信息的输入经过变化以后的输出
    '''
    num_examples = len(X_data)
    total_accuracy = 0
    #生成验证
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        batch_x = ProcessFigs(batch_x)
        accuracy = sess.run(accuracy_operation, feed_dict={x_RGB: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def ProcessFigs(Figs):
    '''
    这个函数用于将输入的图片库文件转换为flot32格式，扩展图片，调换RB，一般化
    '''
    FigOut = np.zeros([np.shape(Figs)[0],227,227,3])
    for Counter in range(np.shape(Figs)[0]):
        AFig = Figs[Counter,:,:,:]
        AFig = skimage.transform.resize(AFig,[227,227],preserve_range = True)
        AFig = AFig - np.mean(AFig)
        AFig[:, :, 0], AFig[:, :, 2] = AFig[:, :, 2], AFig[:, :, 0]
        FigOut[Counter,:,:,:] = AFig[np.newaxis,:,:,:]
    return FigOut
#TODO: Load traffic signs data.
#%%调入存储的数据文件
pickle_file = './train.p'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  
  X = pickle_data['features']
  y = pickle_data['labels']
  
  del pickle_data  # Free up memory
  
# TODO: Split data into training and validation sets.
#%%将数库分裂为训练集和验证集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
del X
del y

#%%创建运行场景
sess = tf.Session()

# TODO: Define placeholders and resize operation.
#%% 设置图片格式转换层
with tf.name_scope('Transform_RGB'):
    x_RGB = tf.placeholder(tf.float64, (None, 227, 227, 3))
    x_RGB = tf.cast(x_RGB,tf.float32)

# TODO: pass placeholder as first argument to `AlexNet`.
#%% 与AlexNet连接，并设置反传信息中断层
with tf.name_scope('AlexNet'):
    fc7 = AlexNet(x_RGB, feature_extract=True)
    #fc7 = AlexNet(resized_x, feature_extract=True)
    # NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
    # past this point, keeping the weights before and up to `fc7` frozen.
    # This also makes training faster, less work to do!
    fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
#%%添加新加入的空白层等待训练并适配网络
with tf.name_scope('New_add1'):
    mu = 0
    sigma = 0.1
    weight8 = tf.Variable(tf.truncated_normal([4096,43],mean = mu, stddev = sigma),name = 'weight8')
    bias8 = tf.Variable(tf.zeros(43),name = 'bias8')
    Mux8 = tf.matmul(fc7,weight8)
    logits8 = tf.add(Mux8,bias8)
    

# TODO: Define loss, training, accuracy operations.
with tf.name_scope('Train_price'):
    y = tf.placeholder(tf.uint8, (None))
    one_hot_y = tf.one_hot(y,43,dtype = tf.float32)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits8)
    loss_operation = tf.reduce_mean(cross_entropy)
    
# TODO: Train and evaluate the feature extraction model.
#%%生成优化器
rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#主要，这里要指定训练的变量
training_operation = optimizer.minimize(loss_operation,var_list = [weight8, bias8])


EPOCHS = 50
BATCH_SIZE = 128

#执行优化过程
init = tf.global_variables_initializer()
sess.run(init)

num_examples = len(X_train)
print("Training...")
print()
for i in range(EPOCHS):
    X_train, y_train = shuffle(X_train, y_train)
    StartTime = time.clock()
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        batch_x = ProcessFigs(batch_x)
        _,Out = sess.run([training_operation,loss_operation], feed_dict={x_RGB: batch_x, y: batch_y})
        print(Out)
    EndTime = time.clock()
    print("这是第{}次训练完成".format(i+1))
    print("这次训练的使用时间：{}".format(EndTime - StartTime))
    print("检测一下准确程度：")
    validation_accuracy = evaluate(X_test,y_test,sess,logits8,one_hot_y,BATCH_SIZE)
    print(validation_accuracy)
    print("~~~~~~~~~~~~~~~~~~~~")
    if validation_accuracy > 0.98:
        break



