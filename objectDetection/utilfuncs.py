import time

from objDetect import *
from consts import *
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def sampledetection():
    cap = get_cap(url)

    ret, frame = cap.read()

    cap.release()
    obj_detect = ObjectDetectionPipeline(device="cuda", threshold=0.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(obj_detect(frame)[:, :, ::-1])
    plt.savefig('/home/shankha/PycharmProjects/imageAnalysis/objectDetection/images/newdetection.jpg')


def sampleframe():
    cap = get_cap(url)

    ret, frame = cap.read()

    cap.release()

    plt.imshow(frame[:, :, ::-1])
    plt.savefig('/home/shankha/PycharmProjects/imageAnalysis/objectDetection/images/new.jpg')


def makevideo() -> None:
    start = time.time()
    batch_size = 16

    cap = get_cap(url)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    size = min([width, height])

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(basepath_video + "out.avi", fourcc, 20, (size, size))

    obj_detect = ObjectDetectionPipeline(device="cuda", threshold=0.5)

    exit_flag = True
    while exit_flag:
        batch_inputs = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if ret:
                batch_inputs.append(frame)
            else:
                exit_flag = False
                break

        outputs = obj_detect(batch_inputs)
        if outputs is not None:
            for output in outputs:
                out.write(output)
        else:
            exit_flag = False

    cap.release()
    totaltime = round((time.time() - start) / 60)
    print(totaltime)


def get_cap(urlname):
    return cv2.VideoCapture(urlname)
