# Import necessary packages
#import threading
#import time

import cv2
import numpy as np
#from play_audioMask import PlayAudio

mask_model_path = 'onnx_mask_detector.onnx'
video_cam_index = 0 

maskModel = cv2.dnn.readNetFromONNX(mask_model_path)
#maskModel.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
stream = cv2.VideoCapture(video_cam_index)

#AudioPlay = False
#playAudio = False

def detect(img, maskModel):
    # Generate a blob
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0 / 255, (128, 128),(0, 0, 0), swapRB=True, crop=False)
    
    maskModel.setInput(blob)
    preds = maskModel.forward()
    prediction_index = np.array(preds)[0].argmax()
    return prediction_index

''''
def thread_for_when_to_play_audio():
    """
    This function is used for playing the alarm if a person is not wearing a mask.
    :return:
    """
    global playAudio
    while True:
        if playAudio:
            play_audio()


def play_audio():
    """
    This function is used for playing the alarm if a person is not wearing a mask.
    :return:
    """
    global AudioPlay
    global playAudio
    SoundThread = threading.Thread(target=PlayAudio.play_audio_file)
    print("[INFO]: Starting Sound Thread")
    if not AudioPlay:
        AudioPlay = True
        SoundThread.start()
        time.sleep(3)
        AudioPlay = False
        playAudio = False
        print("[INFO]: Stopping Sound Thread")
'''

def thread_for_mask_detection():
    global maskModel
    global stream
    global playAudio
    while True:
        # Read frame from the stream
        ret, frame = stream.read()

        # Run the detect function on the frame
        (predictions) = detect(frame, maskModel)

        if predictions == 0:
          print("With Mask")

        elif predictions == 1:
          print("Without Mask")

        else:
          pass

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # break from loop if key pressed is q
        if key == ord("q"):
            break


if __name__ == "__main__":
    #t1 = threading.Thread(target=thread_for_when_to_play_audio)

    #t1.start()

    thread_for_mask_detection()
