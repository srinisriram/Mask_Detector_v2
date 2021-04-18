from threading import Thread
import cv2
from collections import deque


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.frame_list = deque([])
        self.image = None
        self.frame = N

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.image) = self.stream.read()
            self.frame_list.appendright(self.image)

    def read(self):
        self.frame = self.frame_list.popleft()
        return self.frame

    def stop(self):
        self.stopped = True
