import datetime
import glob
from numpy import random
import math
import os
import shutil

import cv2
import numpy as np

from Production.constants import *

'''
Goal of Code: Process
Step #1: Define the class (with_mask or without_mask) that needs to be created into a dataset
Step #2: Load entire Temple_Dataset
Step #3: Open 1st Day
    Step #3a: Open face_images
    Step #3b: Open desired class (defined Step #1)
    Step #3c: For each person, get the total number of images for each person
    Step #3d: Generate a random integer list for the first person (code defined in JamBoard). 
    Step #3e: Iterate through the person using the given random list and store the image paths in a final array for the person. 
    Step #3f: Using the list of paths, move all the images into the final class folder. 
    Step #3g: Clear the lists. 
    Step #3h: Repeat process for next person until all people in Day #1 have been sorted and stored. 
Step #4: Repeat steps #3-3h for every single day until all images in one class have been stored. 
Step #5: Rerun program with new updated class that is the opposite of the previous ran class. 
'''


class Temple_Dataset_Splitting:
    def __init__(self):
        self.desired_class = 'without_mask'
        self.full_temple_dataset = FULL_TEMPLE_DATASET
        self.days_folder = None
        self.persons_path = None
        self.constant_image_path = CONSTANT_IMAGE_PATH
        self.final_dataset = FINAL_DATASET
        self.img_paths = None
        self.img_jpg_paths = []
        self.person_folder = None
        self.random_integers = None
        self.number_of_img = 0

        # 1 = All images, 2 = 50% of images, 3 = 33% of images...
        self.div_constant = 1
        self.debug = True

    def get_all_days_folder(self, path):
        """
        This method sets all the day folders in the var
        """
        self.days_folder = os.listdir(path)
        self.days_folder.sort()
        if self.debug:
            print("[DEBUG][get_all_days_folder] {}".format(path))
            print("[DEBUG][get_all_days_folder] {}".format(self.days_folder))

    def get_all_persons_folder(self, day):
        self.persons_path = self.full_temple_dataset + day + "/" + self.constant_image_path + "/" + self.desired_class
        self.person_folder = os.listdir(self.persons_path)
        self.person_folder.sort()
        if self.debug:
            print("[DEBUG][get_all_persons_folder] {}".format(day))
            print("[DEBUG][get_all_persons_folder] {}".format(self.person_folder))

    def iterate_through_one_person(self, person):
        self.img_paths = self.persons_path + "/" + person + "/" + "*.jpg"
        for i in glob.glob(self.img_paths):
            self.img_jpg_paths.append(str(i))
        if self.debug:
            print("[DEBUG][iterate_through_one_person] {}".format(self.img_paths))
            print("[DEBUG][iterate_through_one_person] {}".format(self.img_jpg_paths))

    def get_length_of_total_img(self, lis):
        self.number_of_img = len(lis)
        if self.debug:
            print("[DEBUG][get_length_of_total_img] {}".format(lis))
            print("[DEBUG][get_length_of_total_img] {}".format(self.number_of_img))

    def get_random_num_list(self, len_list):
        # self.random_integers = random.randint(len_list, size=(math.ceil(len_list / 2)))
        self.random_integers = random.choice(len_list, replace=False, size=(math.ceil(len_list / self.div_constant)))
        if self.debug:
            print("[DEBUG][get_random_num_list] {}".format(len_list))
            print("[DEBUG][get_random_num_list] {}".format(self.random_integers))

    def move_img(self, img_list, random_list):
        if self.debug:
            print("[DEBUG][move_img] {}".format(img_list))
            print("[DEBUG][move_img] {}".format(len(img_list)))
            print("[DEBUG][move_img] {}".format(random_list))
        for i in random_list:
            image_path = img_list[i]
            image_name = image_path.split("/")
            if self.debug:
                print("[DEBUG][move_img] {}".format(image_path))
                print("[DEBUG][move_img] {}".format(image_name))
                print("[DEBUG][move_img] {}".format(image_name[-1]))
                print("[DEBUG][move_img] {}".format(self.final_dataset + self.desired_class + "/" + image_name[-1]))
            shutil.copy(image_path, self.final_dataset + self.desired_class + "/" + image_name[-1])
            
    def clear_files(self):
        self.img_jpg_paths = []
        self.random_integers = None
        self.number_of_img = 0

    def full_iteration(self):
        self.get_all_days_folder(self.full_temple_dataset)
        for i in self.days_folder:
            self.get_all_persons_folder(i)
            for j in self.person_folder:
                self.iterate_through_one_person(j)
                self.get_length_of_total_img(self.img_jpg_paths)
                self.get_random_num_list(self.number_of_img)
                self.move_img(self.img_jpg_paths, self.random_integers)
                self.clear_files()


if __name__ == '__main__':
    temple_data_moving = Temple_Dataset_Splitting()
    temple_data_moving.full_iteration()
