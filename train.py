import transformnet
import extractnet
import numpy as np
import tensorflow as tf
import PIL.Image as Image
import os
from config import *
import uuid
import shutil

def traingen(dirpath):
    flist = os.listdir(dirpath)
    np.random.shuffle(flist)
    cursor = 0
    while cursor<len(flist):
        batchfn = flist[cursor:cursor+batchsize]
        cursor+=batchsize
        ims = []
        for fn in batchfn:
            im = Image.open(os.path.join(dirpath, fn))
            im = im.resize((224, 224), Image.ANTIALIAS)
            im = np.array(im) / 255.
            ims.append(im)
        yield np.array(ims)


sess = tf.Session()
# model1
input = tf.placeholder(dtype='float32', shape=(None, None, None, 3),name="input")
synthesizetensor = transformnet.transform(input)
predcontenttensor, predstyletensors = extractnet.extract(synthesizetensor,prefix="A")
# model2
contenttensor, styletensors = extractnet.extract(input,prefix="B")

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
    gram = np.matmul(style,styletranspose)/style.shape[1]
    grams.append(gram)
loss = 7.5*tf.reduce_sum((predcontenttensor-contenttensor)**2)/tf.cast(tf.size(contenttensor),dtype=tf.float32)
for predstyletensor,truthgram in zip(predstyletensors,grams):
    predstyletensor = tf.transpose(predstyletensor,[0,3,1,2])
    shape = tf.shape(predstyletensor)
    predstyletensor = tf.reshape(predstyletensor,shape=[shape[0],shape[1],-1])
    grampred = tf.matmul(predstyletensor,predstyletensor,transpose_b=True)/tf.cast(tf.shape(predstyletensor)[2],dtype=tf.float32)
    loss+=100*tf.reduce_sum((grampred-truthgram)**2)/truthgram.size
loss/=batchsize
saver = tf.train.Saver()
trainop = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess.run(tf.global_variables_initializer())

def train():
    for ep in range(epoch):
        testims = next(traingen(testpath))
        minerr = 999999999
        argminerr = -1
        errorrec = []
        for i, batchims in enumerate(traingen(trainpath)):
            _, losstrain = sess.run([trainop, loss], feed_dict={input: batchims})
            print(losstrain)
            if i % 5 == 0:
                synthesizeimgs, lossval = sess.run([synthesizetensor, loss], feed_dict={input: testims})
                errorrec.append(lossval)
                if len(errorrec) - argminerr > 30:
                    return
                list(map(os.unlink, (os.path.join(savepath, f) for f in os.listdir(savepath))))
                for im in synthesizeimgs:
                    im = np.uint8(im*255)
                    Image.fromarray(im).save(os.path.join(savepath, str(uuid.uuid4()) + ".jpg"))
                if lossval < minerr:
                    minerr = lossval
                    argminerr = len(errorrec) - 1
                    shutil.rmtree(modelpath,True)
                    saver.save(sess,modelpath)

train()






# tf.summary.scalar('cc',loss)
# merged = tf.summary.merge_all()
# train_writer = tf.summary.FileWriter(r"D:\Users\yl_gong\Desktop\log",
#                                       sess.graph)
# sess.run(tf.global_variables_initializer())
# im = Image.open(testimg)
# im = np.expand_dims(np.array(im)/255.,axis=0)
# summary = sess.run(merged,feed_dict={input:im})
# train_writer.add_summary(summary, 0)