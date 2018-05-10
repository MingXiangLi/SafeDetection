import os
import cv2
import time
import numpy as np
import tensorflow as tf

from utils.app_utils import FPS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()
#调用自己下载好的模型
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)

category_index = label_map_util.create_category_index(categories)

def detect_objects(image_np, sess, detection_graph,inceptionsess):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    # 这里的class是包含多个识别种类的二维数组
    #[[100,4]]boxes 每个框的位置坐标,    scores 100个 ,     classes 100个 ,    num_detections 100个
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        inceptionsess,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.6
        )
    return image_np


if __name__ == '__main__':
    #tf.Graph()生成新的图
    detection_graph = tf.Graph()
    inceptionsess =tf.Graph()
    with inceptionsess.as_default():
        od_graph_def = tf.GraphDef()
        #在当前路径下配置好inception训练出的模型
        with tf.gfile.FastGFile('output_graph.pb', 'rb') as f:
            serialized_graph = f.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
    video_capture = cv2.VideoCapture('a.flv')
    fps = FPS().start()
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))

    out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        t = time.time()
        detected_image = detect_objects(frame, sess, detection_graph,inceptionsess)
        fps.update()
        #out.write(detected_image)
        cv2.imshow('Video', detected_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()
    video_capture.release()
    sess.close()
    cv2.destroyAllWindows()
