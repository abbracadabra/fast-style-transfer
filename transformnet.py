import numpy as np
import tensorflow as tf

def transform(input):
    #inp = tf.placeholder(dtype='float32',shape=[None,size[0],size[1],3])
    tmp = conv(input,filternum = 32,kernelsize =9,strides = 1)
    tmp = conv(tmp, filternum=64, kernelsize=3, strides=2)
    tmp = conv(tmp, filternum=128, kernelsize=3, strides=2)
    tmp = resblock(tmp, filternum=128,kernelsize=3)
    tmp = resblock(tmp, filternum=128, kernelsize=3)
    tmp = resblock(tmp, filternum=128, kernelsize=3)
    tmp = resblock(tmp, filternum=128, kernelsize=3)
    tmp = resblock(tmp, filternum=128, kernelsize=3)
    tmp = convtranspose(tmp,filters=64,kernel_size=3,strides=2)
    tmp = convtranspose(tmp, filters=32, kernel_size=3, strides=2)
    tmp = conv(tmp, filternum=3, kernelsize=9, strides=1,relu=False)
    tmp = tf.nn.sigmoid(tmp)
    output = tf.identity(tmp,name="output")
    return output

def resblock(inp,filternum,kernelsize):
    tmp = conv(inp,filternum,kernelsize,1)
    tmp = conv(inp, filternum, kernelsize,1,relu=False)
    tmp = tf.nn.leaky_relu(inp+tmp)
    return tmp

def conv(inp,filternum,kernelsize,strides,relu=True):
    tmp = tf.layers.conv2d(inp, filters=filternum, kernel_size=kernelsize, strides=strides, padding='SAME')
    normalized = instancenormalize(tmp)
    tmp = tf.contrib.layers.bias_add(normalized)
    if relu:
        tmp = tf.nn.leaky_relu(tmp)
    return tmp

def instancenormalize(inp):
    mu, var = tf.nn.moments(inp, axes=[1, 2], keep_dims=True)
    normalized = (inp - mu) / tf.sqrt(var + 1e-5)
    return normalized

def convtranspose(inp,filters,kernel_size,strides):
    tmp = tf.layers.conv2d_transpose(inp, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME')
    normalized = instancenormalize(tmp)
    tmp = tf.contrib.layers.bias_add(normalized)
    tmp = tf.nn.leaky_relu(tmp)
    return tmp







