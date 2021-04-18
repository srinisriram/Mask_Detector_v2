from threading import Thread
import cv2


class Threaded_Video_Stream:
    def __init__(self):
        print("Starting __init__")
        self.src = 0
        self.stream = cv2.VideoCapture(self.src)
        (self.grabbed, self.frame) = self.stream.read()
        print("Printing self.frame", self.frame)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        print("Starting start function")
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        # return self

    def update(self):
        print("Starting update function")
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        print("Starting read function")
        # return the frame most recently read
        print("Printing self.frame:\n", self.frame)
        return self.frame

    def stop(self):
        print("Starting stop function")
        # indicate that the thread should be stopped
        self.stopped = True
