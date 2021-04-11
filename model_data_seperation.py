import datetime
import glob
from numpy import random
import math
import os
import shutil

import cv2
import imutils
import numpy as np

from Production.constants import *
from capture_images import Capture_Images
from temple_dataset_splitting import Temple_Dataset_Splitting


class BlurredSeparation:

    def __init__(self):
        self.cv2_image = None
        self.class_name = 'without_mask'
        self.files = None
        self.folder_path = None
        self.trash_file = None
        self.destination = None
        self.num_image = None
        self.full_file_path = None
        self.super_res_image = None
        self.sr = None

    def get_files(self):
        self.folder_path = NEW_DATA_PATH + self.class_name
        self.files = os.listdir(self.folder_path)
        print(len(self.files))

    def move_files(self, file):
        self.trash_file = file
        self.destination = TRASH_IMAGES
        shutil.move(self.trash_file, self.destination)

    def full_iteration(self):
        self.get_files()
        count = 0
        for i in self.files:
            self.full_file_path = '/home/srinivassriram/Desktop/Mask_Detector_v2/temple_final_dataset/without_mask/' + i
            self.num_image = cv2.imread(self.full_file_path)
            if Capture_Images().is_blur(self.num_image, MIN_THRESHOLD):
                self.move_files(self.full_file_path)
                count = count + 1
            else:
                continue
        print("Num of Files Deleted:", count)


if __name__ == "__main__":
    BlurredSeparation().full_iteration()
