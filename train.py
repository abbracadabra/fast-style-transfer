import transformnet
import extractnet
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import os
from config import *

def traingen():
    flist = os.listdir(trainpath)
    flist = np.random.shuffle(flist)
    cursor = 0
    while cursor+batchsize<=len(flist):
        batchfn = flist[cursor:cursor+batchsize]
    ims = []
    for fn in batchfn:
        im = Image.open(os.path.join(trainpath,fn))
        im = im.resize((224, 224),Image.ANTIALIAS)
        im = np.array(im) / 255.
        ims.append(im)
    yield ims

sess = tf.Session()
# model1
input = tf.placeholder(dtype='float32', shape=(None, None, None, 3),name="input")
trans_out = transformnet.transform(input)
predcontenttensor, predstyletensors = extractnet.extract(trans_out,prefix="A")
# model2
contenttensor, styletensors = extractnet.extract(input,prefix="B")
sess.run(tf.global_variables_initializer())
styleim = Image.open(styleimg)
styleim = styleim.resize((224, 224), Image.ANTIALIAS)
styleim = np.expand_dims(np.array(styleim) / 255., axis=0)
styles = sess.run(styletensors,feed_dict={input:styleim})
grams = []
for style in styles:
    style = np.squeeze(style)
    style = np.transpose(style,[2,0,1])
    channelnum = style.shape[0]
    style = np.resize(style,(channelnum,-1))
    styletranspose = np.transpose(style,[1,0])
    gram = np.matmul(style,styletranspose)
    grams.append(gram)
loss = tf.reduce_sum((predcontenttensor-contenttensor)**2)
for predstyletensor,styletensor in zip(predstyletensors,styletensors):
    predstyletensor = tf.transpose(predstyletensor,[0,3,1,2])
    shape = tf.shape(predstyletensor)
    predstyletensor = tf.reshape(predstyletensor,shape=[shape[0],shape[1],-1])
    grama = tf.matmul(predstyletensor,predstyletensor,transpose_b=True)
    styletensor = tf.transpose(styletensor,[0,3,1,2])
    shape = tf.shape(styletensor)
    styletensor = tf.reshape(styletensor, shape=[shape[0], shape[1], -1])
    gramb = tf.matmul(styletensor,styletensor,transpose_b=True)
    loss = loss+tf.reduce_sum((grama-gramb)**2)
trainop = tf.train.AdamOptimizer(loss)

for ep in range(epoch):
    for batchim in traingen():
        _,lossval = sess.run([trainop,loss],feed_dict={input:batchim})

