import numpy as np
import tensorflow as tf

# aa = tf.constant([[[[1.],[1.]]]])
#
# # bb = tf.nn.conv2d_transpose(aa,
# #     [[[[1.]],[[2.]],[[3.]]],[[[4.]],[[5.]],[[6.]]],[[[7.]],[[8.]],[[9.]]]],
# #     [1,3,3,1],
# #     [1,1,1,1],
# #     padding="VALID",
# #     data_format="NHWC")
#
# bb = tf.layers.conv2d_transpose(aa,filters=1,kernel_size=3,strides=2,padding='SAME')
# tf.set_random_seed(1)
# sess = tf.Session()
# sess.run([tf.global_variables_initializer()])
# print(sess.run([bb]))

