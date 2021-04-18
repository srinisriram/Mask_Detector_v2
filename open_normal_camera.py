import numpy as np
import cv2
from imutils.video import FPS

cap = cv2.VideoCapture(0)
fps = FPS().start()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    fps.update()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cap.release()
cv2.destroyAllWindows()