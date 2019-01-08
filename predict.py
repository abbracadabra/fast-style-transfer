import numpy as np
import tensorflow as tf
import PIL.Image as Image
import os
from config import *
import PIL.Image

saver = tf.train.import_meta_graph(modelpath)
sess = tf.Session()
saver.restore(sess,modelpath)
input = tf.get_default_graph().get_tensor_by_name("input:0")
output = tf.get_default_graph().get_tensor_by_name("output:0")

im = Image.open(testimg)
im = np.expand_dims(np.array(im)/255.,axis=0)
im = sess.run([output],feed_dict={input:im})
im = Image.fromarray(np.squeeze(im))
im.show()