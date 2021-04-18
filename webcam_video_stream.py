# import the necessary packages
from threading import Thread
import cv2
from collections import deque


class WebcamVideoStream:
    def __init__(self, src=0, name="WebcamVideoStream"):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.image) = self.stream.read()
        self.frame_list = deque([])
        self.frame = None

        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.image) = self.stream.read()
            self.frame_list.append(self.image)

    def read(self):
        if len(self.frame_list) != 0:
            self.frame = self.frame_list.popleft()
            return self.frame
        else:
            pass

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
