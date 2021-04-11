import datetime
import os
import threading
import time
import sys

import cv2
import imutils
import numpy as np
from Production.constants import *
from Production.play_audio import PlayAudio
from Production.frames_per_second import FPS
from imutils.video import VideoStream


class Capture_Images:
    run_program = RUN_PROGRAM
    input_video_file_path = None
    preferable_target = cv2.dnn.DNN_TARGET_CPU

    def __init__(self):
        self.SoundThread = None
        self.AudioPlay = None
        self.mask_model = None
        self.frame = None
        self.h = None
        self.w = None
        self.vs = None
        self.image_blob = None
        self.confidence = None
        self.detections = None
        self.box = None
        self.face = None
        self.f_h = None
        self.f_w = None
        self.startX = None
        self.startY = None
        self.endX = None
        self.endY = None
        self.face_blob = None
        self.predictions = None
        self.name = None
        self.detector = None
        self.prediction_index = None
        self.fileName = None
        self.text = None
        self.y = None
        self.colorIndex = None
        self.threshold = MIN_THRESHOLD
        self.model_input_size = MODEL_INPUT_SIZE
        self.crash_log = CRASH_LOG
        self.current_time = None
        self.time = ""
        self.seconds = None
        self.debug = False

        self.fps_instance = FPS()
        self.fps_instance.start()
        self.fps_instance.stop()
        self.load_caffe_model()
        self.load_pytorch_model()
        self.initialize_camera()

    @classmethod
    def perform_job(cls, preferableTarget=preferable_target, input_video_file_path=input_video_file_path):
        """
        This method performs the job expected from this class.
        :key
        """
        # Set preferable target.
        Capture_Images.preferable_target = preferableTarget
        # Set input video file path (if applicable)
        Capture_Images.input_video_file_path = input_video_file_path
        # Create a thread that uses the thread_for_mask_detection function and start it.
        t1 = threading.Thread(target=Capture_Images().thread_for_mask_detection)
        t1.start()

    def is_blur(self, frame, thresh):
        fm = cv2.Laplacian(frame, cv2.CV_64F).var()
        if fm < thresh:
            return True
        else:
            return False

    def super_res(self, frame):
        frame = cv2.resize(frame, (self.model_input_size, self.model_input_size), interpolation=cv2.INTER_CUBIC)

    def load_caffe_model(self):
        """
        This function will load the caffe model that we will use for detecting a face, and then set the preferable target to the correct target.
        :key
        """
        print("Loading caffe model used for detecting a face.")

        # Use cv2.dnn function to read the caffe model used for detecting faces and set preferable target.
        self.detector = cv2.dnn.readNetFromCaffe(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            prototxt_path),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                model_path))

        self.detector.setPreferableTarget(Capture_Images.preferable_target)

    def load_pytorch_model(self):
        """
        This function will load the pytorch model that is used for predicting the class of the face.
        :key
        """
        print("Loading Mask Detection Model")

        self.mask_model = cv2.dnn.readNetFromONNX(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            mask_model_path))

        self.mask_model.setPreferableTarget(Capture_Images.preferable_target)

    def initialize_camera(self):
        """
        This function will initialize the camera or video stream by figuring out whether to stream the camera capture or from a video file.
        :key
        """
        if Capture_Images.input_video_file_path is None:
            print("[INFO] starting video stream...")
            self.vs = VideoStream(src=VID_CAM_INDEX).start()
            print("[INFO] Waiting 2 Seconds...")
            time.sleep(SLEEP_TIME_AMOUNT)
        else:
            self.vs = cv2.VideoCapture(Capture_Images.input_video_file_path)

    def grab_next_frame(self):
        """
        This function extracts the next frame from the video stream.
        :return:
        """
        self.fps_instance.update()
        if Capture_Images.input_video_file_path is None:
            self.frame = self.vs.read()
        else:
            _, self.frame = self.vs.read()
            # self.frame = cv2.rotate(self.frame, cv2.ROTATE_180)

        if self.frame is None:
            return
        self.frame = imutils.resize(self.frame, width=frame_width_in_pixels)

    def set_dimensions_for_frame(self):
        """
        This function will set the frame dimensions, which we will use later on.
        :key
        """
        if not self.h or not self.w:
            # Set frame dimensions.
            (self.h, self.w) = self.frame.shape[:2]

    def create_frame_blob(self):
        """
        This function will create a blob for our face detector to detect a face.
        :key
        """
        self.image_blob = cv2.dnn.blobFromImage(
            cv2.resize(self.frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

    def extract_face_detections(self):
        """
        This function will extract each face detection that our face detection model provides.
        :return:
        """

        self.detector.setInput(self.image_blob)
        self.detections = self.detector.forward()

    def extract_confidence_from_face_detections(self, i):
        """
        This function will extract the confidence(probability) of the face detection so that we can filter out weak detections.
        :param i:
        :return:
        """
        self.confidence = self.detections[0, 0, i, 2]

    def create_face_box(self, i):
        """
        This function will define coordinates of the face.
        :param i:
        :return:
        """
        self.box = self.detections[0, 0, i, 3:7] * np.array([self.w, self.h, self.w, self.h])
        (self.startX, self.startY, self.endX, self.endY) = self.box.astype("int")

    def save_img(self, frame, folder_path):
        """
        This function will save the frame or the roi into the folder
        :param frame:
        :param folder_path:
        :return:
        """
        self.fileName = folder_path + str(datetime.datetime.now().timestamp()) + ".jpg"
        cv2.imwrite(filename=self.fileName, img=frame)

    def extract_face_roi(self):
        """
        This function will use the coordinates defined earlier and create a ROI that we will use for embeddings.
        :return:
        """
        self.face = self.frame[self.startY:self.endY, self.startX:self.endX]
        (self.f_h, self.f_w) = self.face.shape[:2]

    def create_predictions_blob(self):
        """
        This function will create another blob out of the face ROI that we will use for prediction.
        :return:
        """
        print("Create predictions blob")
        self.face_blob = cv2.dnn.blobFromImage(cv2.resize(self.face,
                                                          (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), 1.0 / 255,
                                               (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), (0, 0, 0),
                                               swapRB=True, crop=False)

    def extract_detections(self):
        """
        This function uses the PyTorch model to predict from the given face blob.
        :return:
        """
        self.mask_model.setInput(self.face_blob)
        self.predictions = self.mask_model.forward()

    def perform_classification(self):
        """
        This function will now use the prediction to do the following:
            1. Extract the class prediction from the predictions.
            2. Get the label of the prediction.
        :return:
        """
        self.prediction_index = np.array(self.predictions)[0].argmax()
        if self.prediction_index == MASK_INDEX:
            self.name = "With Mask"
        elif self.prediction_index == NO_MASK_INDEX:
            self.name = "Without Mask"
        else:
            pass

    def play_audio(self):
        """
        This function is used for playing the alarm if a person is not wearing a mask.
        :return:
        """
        self.SoundThread = threading.Thread(target=PlayAudio.play_audio_file)
        print("[INFO]: Starting Sound Thread")
        if not self.AudioPlay:
            self.AudioPlay = True
            self.SoundThread.start()
            time.sleep(3)
            self.AudioPlay = False
            print("[INFO]: Stopping Sound Thread")

    def create_frame_icons(self):
        """
        This function will create the icons that will be displayed on the frame.
        :return:
        """
        self.text = "{}".format(self.name)
        self.y = self.startY - 10 if self.startY - 10 > 10 else self.startY + 10
        self.colorIndex = LABELS.index(self.name)

    def loop_over_frames(self):
        """
        This is the main function that will loop through the frames and use the functions defined above to detect for face mask.
        :return:
        """
        while Capture_Images.run_program:
            self.grab_next_frame()
            self.set_dimensions_for_frame()
            self.create_frame_blob()
            self.extract_face_detections()

            for i in range(0, self.detections.shape[2]):
                self.extract_confidence_from_face_detections(i)
                if self.confidence > MIN_CONFIDENCE:
                    self.create_face_box(i)
                    self.extract_face_roi()
                    if self.f_w < 20 or self.f_h < 20:
                        continue
                    if self.is_blur(self.face, self.threshold):
                        continue
                    else:
                        self.super_res(self.face)
                        self.create_predictions_blob()
                        self.extract_detections()
                        self.perform_classification()
                        if self.name == "With Mask":
                            print("[Prediction] Person is wearing a mask.")
                            self.save_img(frame=self.frame, folder_path=mask_full_folder)
                            self.save_img(frame=self.face, folder_path=mask_face_folder)
                        if self.name == "Without Mask":
                            print("[Prediction] Person is not wearing a mask.")
                            self.save_img(frame=self.face, folder_path=without_mask_face_folder)
                            self.save_img(frame=self.frame, folder_path=without_mask_full_folder)
                            self.play_audio()
                        self.create_frame_icons()
                        cv2.rectangle(self.frame, (self.startX, self.startY), (self.endX, self.endY),
                                      COLORS[self.colorIndex], 2)
                        cv2.putText(self.frame, self.text, (self.startX, self.y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    COLORS[self.colorIndex], 2)
            if OPEN_DISPLAY:
                print("[FPS] ", self.fps_instance.fps)

                cv2.imshow("mask_detector_frame", self.frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

    def clean_up(self):
        """
        Clean up the cv2 video capture.
        :return:
        """
        cv2.destroyAllWindows()
        self.vs.release()
        self.fps_instance.stop()
        print("elapsed time: {:.2f}".format(self.fps_instance.elapsed()))
        print("approx. FPS: {:.2f}".format(self.fps_instance.fps()))

    def get_current_time(self):
        self.current_time = str(datetime.datetime.now()).split(" ")
        self.current_time = self.current_time[1].split(":")
        self.seconds = self.current_time[2].split(".")[0]
        self.current_time.pop(2)
        for i in self.current_time:
            self.time += ":" + i
        self.time += ":" + self.seconds
        self.current_time = self.time[1:]
        if self.debug:
            print(self.seconds)
            print(self.time)
            print(self.current_time)
        return self.current_time

    def write_crash_to_file(self, exception_msg):
        crash_log_file_object = open(self.crash_log, 'a')
        time = self.get_current_time()
        crash_msg = str(time) + "-" + str(exception_msg) + "\n"
        crash_log_file_object.write(crash_msg)
        crash_log_file_object.close()

    def thread_for_mask_detection(self):
        """
        Callable function that will run the mask detector and can be invoked in a thread.
        :return:
        """
        while Capture_Images.run_program:
            try:
                self.loop_over_frames()
            except Exception:
                self.write_crash_to_file(sys.exc_info()[0])
            finally:
                self.clean_up()


if __name__ == "__main__":
    Capture_Images.perform_job(preferableTarget=cv2.dnn.DNN_TARGET_MYRIAD)
