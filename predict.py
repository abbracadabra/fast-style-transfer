import numpy as np
import tensorflow as tf
import PIL.Image as Image
import os
from config import *

saver = tf.train.import_meta_graph(modelpath)
sess = tf.Session()
saver.restore(sess,modelpath)
input = tf.get_default_graph().get_tensor_by_name("input:0")
output = tf.get_default_graph().get_tensor_by_name("output:0")

im = Image.open(testimg)
im = im.resize((224,224), Image.ANTIALIAS)
im = np.expand_dims(np.array(im)/255.,axis=0)
im = sess.run([output],feed_dict={output:im})
im.show()