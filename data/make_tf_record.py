import tensorflow as tf
import random
import math
import sys
import os

_DATA_DIR = "/home/user01/Desktop/airi400-project-deepfashionet/data/Img"
# 총 28만개의 데이터 중 20%는 test data, 나머지의 20% 데이터는 validation data
_NUM_VALIDATION = 44800
_RANDOM_SEED = 0
# 5개로 나눠서 구성(cross validation 가능)
_NUM_SHARDS = 5


# image의 차원과 decode를 할 클래스를 생성
class ImageReader(object):
    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self._decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})

        # image shape의 크기가 3이 아니거나 채널이 3이 아니면 표시
        assert (len(image.shape) == 3)
        assert (image.shape[2] == 3)
        return image


