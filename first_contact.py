import os, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

t = tf.constant([[1,2,3],[4,5,6]])
print(t)
print(t.shape)