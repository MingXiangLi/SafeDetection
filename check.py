import tensorflow as tf
import numpy as np
from pylab import array

def check(image,left, right, top, bottom,inceptionsess):
    got = array(image)
    crop_img = got[int(top):int(bottom), int(left):int(right), 0:3]
    with tf.Session(graph=inceptionsess) as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        # jpeg 进行编码
        # """Return the value of the tensor represented by this handle.""
        encode = tf.image.encode_jpeg(crop_img)
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': encode.eval()})  # 图片格式是jpg格式
        predictions = np.squeeze(predictions)  # 把结果转为1维数据
        top_k = predictions.argsort()[::-1]
        if top_k[0]==1:
            human_string="unsafe"
        else:
            human_string="safe"
        return human_string

