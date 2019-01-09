import numpy as np
import tensorflow as tf
import tensorflow.gfile as gfile
import PIL.Image as Image

# dd = tf.import_graph_def(r"D:\Users\yl_gong\Desktop\dl\mobilenet_v2_0.35_224\mobilenet_v2_0.35_224_frozen.pb")
# print(dd)

im = Image.open(r'D:\Users\yl_gong\Desktop\dl\voc\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000542.jpg')
im = im.resize((224,224), Image.ANTIALIAS)
im = np.expand_dims(np.array(im)/255.,axis=0)

with gfile.FastGFile(r"D:\Users\yl_gong\Desktop\dl\mobilenet_v2_0.35_224\mobilenet_v2_0.35_224_frozen.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def)

input = tf.get_default_graph().get_tensor_by_name("import/input:0")
output = tf.get_default_graph().get_tensor_by_name("import/MobilenetV2/Predictions/Reshape_1:0")




