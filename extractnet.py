import numpy as np
import tensorflow as tf
import tensorflow.gfile as gfile
import PIL.Image as Image
from config import *

# im = Image.open(r'D:\Users\yl_gong\Desktop\dl\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000542.jpg')
# im = im.resize((224,224), Image.ANTIALIAS)
# im = np.expand_dims(np.array(im)/255.,axis=0)



def extract(input,prefix):
    with gfile.FastGFile(r"D:\Users\yl_gong\Desktop\dl\mobilenet_v2_0.35_224\mobilenet_v2_0.35_224_frozen.pb",
                         'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def,input_map={"input:0": input},name=prefix)
        #print(tf.get_default_graph().get_operation_by_name("import/input"))
        # for i in tf.get_default_graph().get_operations():
        #     print(i.name)

    #input = tf.get_default_graph().get_tensor_by_name(prefix+"/input:0")
    contenttensor = tf.get_default_graph().get_tensor_by_name(prefix+"/"+content)
    styles = []
    for t in styletensors:
        styles.append(tf.get_default_graph().get_tensor_by_name(prefix+"/"+t))
    return contenttensor,styles
    # input = tf.get_default_graph().get_tensor_by_name("input:0")
    # output = tf.get_default_graph().get_tensor_by_name("import/MobilenetV2/Predictions/Reshape_1:0")
    # sess = tf.Session()
    # print(sess.run(output,feed_dict={input:im}))

#extract(tf.placeholder(dtype='float32', shape=(None, 224, 224, 3),name="input"))