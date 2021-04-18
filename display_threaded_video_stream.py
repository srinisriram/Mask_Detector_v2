from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2

display = True
run_program = True
print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
while run_program:
    frame = vs.read()
    frame = imutils.resize(frame, width=320)
    if display:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        run_program = False
    fps.update()
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
