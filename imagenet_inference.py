# NOTE: You don't need to edit this code.
import time
import tensorflow as tf
import numpy as np

#from scipy.misc import imread这个属于SCI_Py
from imageio import imread

#这个是一千只够的种类名称
from caffe_classes import class_names

#这个是alexnet的定义函数
from alexnet import AlexNet

# placeholders
x = tf.placeholder(tf.float32, (None, 227, 227, 3))

# By keeping `feature_extract` set to `False`
# we indicate to keep the 1000 class final layer
# originally used to train on ImageNet.
probs = AlexNet(x, feature_extract=False)

# Read Images
im1 = (imread(".\Example_Figs\weasel.png")[:, :, :3]).astype(np.float32)
im1 = im1 - np.mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread(".\Example_Figs\poodle.png")[:, :, :3]).astype(np.float32)
im2 = im2 - np.mean(im2)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]

im3 = (imread(".\Example_Figs\dog.png")[:, :, :3]).astype(np.float32)
im3 = im3 - np.mean(im3)
im3[:, :, 0], im3[:, :, 2] = im3[:, :, 2], im3[:, :, 0]

im4 = (imread(".\Example_Figs\dog2.png")[:, :, :3]).astype(np.float32)
im4 = im4 - np.mean(im4)
im4[:, :, 0], im4[:, :, 2] = im4[:, :, 2], im4[:, :, 0]

im5 = (imread(".\Example_Figs\quail.jpg")[:, :, :3]).astype(np.float32)
im5 = im5 - np.mean(im5)
im5[:, :, 0], im5[:, :, 2] = im5[:, :, 2], im5[:, :, 0]

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#%%
# Run Inference
t = time.time()
output = sess.run(probs, feed_dict={x: [im1, im2, im3, im4, im5]})

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()

print("Time: %.3f seconds" % (time.time() - t))
