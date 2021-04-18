import imutils
import time
import cv2
from threaded_video_capture import Threaded_Video_Stream
from Production.frames_per_second import FPS

print("[INFO] Started Video Stream")
display = True
vs = Threaded_Video_Stream().start()
fps = FPS().start()

while True:
    frame = vs.read()
    frame = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_CUBIC)
    if display:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    fps.update()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
