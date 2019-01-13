import numpy as np
import tensorflow as tf

aa = tf.matmul([[1],[2]],[[1],[2]],transpose_b=True)

print(tf.Session().run(aa))

