import os
import cv2
import time
from utils.app_utils import FPS

CWD_PATH = os.getcwd()
if __name__ == '__main__':
    video_capture = cv2.VideoCapture('outpy.mp4')
    fps = FPS().start()
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter('outpy2.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        t = time.time()
        detected_image = frame
        fps.update()
        out.write(detected_image)
        cv2.imshow('Video', detected_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    fps.stop()
    video_capture.release()
    cv2.destroyAllWindows()
