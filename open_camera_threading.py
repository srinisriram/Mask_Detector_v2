from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2
from threaded_video_capture import Threaded_Video_Stream
from Production.frames_per_second import FPS

print("Setting up Video Stream")
display = True
vs = Threaded_Video_Stream().start()
time.sleep(1.0)
fps = FPS().start()

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=320)
	fps.update()
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	if display:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()