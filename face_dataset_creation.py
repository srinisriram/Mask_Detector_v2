import datetime
import glob
import shutil
import os

import cv2
import numpy as np

from Production.constants import *


class Face_Dataset:

    def __init__(self):
        self.run_program = True
        self.current_file = None
        self.all_img_path = ALL_IMAGES_PATH
        self.files = []
        self.saving_img_dir = SAVE_IMAGES_PATH
        self.img = None
        self.h = None
        self.w = None
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
        self.name = None
        self.detector = None
        self.fileName = None

        # Initialize the program and load the necessary files
        self.init_files(self.all_img_path)
        self.load_caffe_model()

    def init_files(self, path):
        """
         This function takes in a path and loads all of the images to a local list using the glob package.
         :param path:
         :return:
        """
        for i in glob.glob(path):
            self.files.append(str(i))

    def load_caffe_model(self):
        """
        This function will load the caffe model that we will use for detecting a face.
        :key
        """
        print("Loading caffe model used for detecting a face.")
        self.detector = cv2.dnn.readNetFromCaffe(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            prototxt_path),
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                model_path))

    def read_image(self, path):
        """
        This function takes the path of an image and opens it using opencv into a variable.
        :param path:
        :return:
        """
        self.img = cv2.imread(path)

    def set_dimensions_for_frame(self):
        """
        This function will set the frame dimensions, which we will use later on.
        :key
        """
        if not self.h or not self.w:
            # Set frame dimensions.
            (self.h, self.w) = self.img.shape[:2]

    def create_frame_blob(self):
        """
        This function will create a blob for our face detector to detect a face.
        :key
        """
        self.image_blob = cv2.dnn.blobFromImage(
            cv2.resize(self.img, (300, 300)), 1.0, (300, 300),
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

    def extract_face_roi(self):
        """
        This function will use the coordinates defined earlier and create a ROI that we will use for embeddings.
        :return:
        """
        self.face = self.img[self.startY:self.endY, self.startX:self.endX]
        (self.f_h, self.f_w) = self.face.shape[:2]

    def save_img(self, img, folder_path, name):
        """
        This function will save the face roi to a folder
        :param frame:
        :param folder_path:
        :return:
        """
        self.fileName = folder_path + name + str(datetime.datetime.now().timestamp()) + ".jpg"
        cv2.imwrite(filename=self.fileName, img=img)
        shutil.move(self.fileName,
                    "temple_final_dataset/without_mask/" + name + str(datetime.datetime.now().timestamp()) + ".jpg")

    def get_file_count(self, path):
        return len(os.listdir(path))

    def perform_job(self):
        """
        This method performs the job expected from this class.
        :key
        """
        for self.current_file in self.files:
            file_count = self.get_file_count("temple_final_dataset/without_mask")
            if file_count <= 1786:
                print(self.current_file)
                self.read_image(self.current_file)
                self.set_dimensions_for_frame()
                self.create_frame_blob()
                self.extract_face_detections()

                for i in range(0, self.detections.shape[2]):
                    self.extract_confidence_from_face_detections(i)
                    if self.confidence > MIN_CONFIDENCE_DATASET:
                        self.create_face_box(i)
                        self.extract_face_roi()
                        if self.f_w < 20 or self.f_h < 20:
                            print("Bad Image: {}".format(self.current_file))
                            continue
                        else:
                            print("Saving Image: {}, {}".format(self.fileName, self.confidence))
                            self.save_img(self.face, self.saving_img_dir, "with_mask")
                            if OPEN_DISPLAY:
                                cv2.imshow("Images Going Over", self.face)
                                key = cv2.waitKey(1) & 0xFF
                                if key == ord("q"):
                                    break
            else:
                break
        print("Finished Iterating over all of the images.")
        self.clean_up()

    def clean_up(self):
        """
        Clean up the cv2 video capture.
        :return:
        """
        print("Cleaning up...")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    face_dataset = Face_Dataset()
    face_dataset.perform_job()
