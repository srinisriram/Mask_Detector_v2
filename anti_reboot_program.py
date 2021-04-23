import datetime
from datetimerange import DateTimeRange
from capture_images import Capture_Images
from Production.constants import *
from crash_email_sending import CrashReport
import os

class Anti_Reboot:
    def __init__(self):
        self.debug = False
        self.seconds = None
        self.time_now = None
        self.capture_inst = Capture_Images()
        self.five_minutes_ago_time = None
        self.old_seconds = None
        self.time = ""
        self.time_range = None
        self.crash_file = None
        self.formatted_time = None
        self.times_list = []
        self.len_file = None
        self.bool = None
        self.extracted_time = None
        self.bool_counter = 0
        self.max_num_of_times_bf_reboot = MAX_NUM_OF_TIMES_BF_REBOOT
        self.constants_file_path = CONSTANTS_FILE

    def get_current_time(self):
        self.time_now = self.capture_inst.get_current_time()
        if self.debug:
            print(self.time_now)

    def get_five_minutes_ago(self):
        self.five_minutes_ago_time = str(datetime.datetime.now() - datetime.timedelta(minutes=5)).split(" ")
        self.five_minutes_ago_time = self.five_minutes_ago_time[1].split(":")
        self.old_seconds = self.five_minutes_ago_time[2].split(".")[0]
        self.five_minutes_ago_time.pop(2)
        for i in self.five_minutes_ago_time:
            self.time += ":" + i
        self.time += ":" + self.old_seconds
        self.five_minutes_ago_time = self.time[1:]
        if self.debug:
            print(self.five_minutes_ago_time)

    def get_time_range(self, old_time, new_time):
        self.time_range = DateTimeRange(old_time, new_time)
        if self.debug:
            print(self.time_range)

    def loop_through_text_file(self):
        self.crash_file = open(CRASH_LOG, "r")
        for i in self.crash_file:
            self.formatted_time = i.split('-')[0]
            self.times_list.append(self.formatted_time)
        if self.debug:
            print(self.times_list)
        self.crash_file.close()

    def loop_through_constants_file(self, content="RUN_PROGRAM = False\n"):
        with open(self.constants_file_path, "r") as self.constants_file:
            lines = self.constants_file.readlines()
        with open(self.constants_file_path, "w") as self.constants_file:
            for line in lines:
                if line.strip("\n") != "RUN_PROGRAM = True":
                    self.constants_file.write(line)
        with open(self.constants_file_path, "r") as self.constants_file:
            with open('constants_copy.txt', 'w') as f:
                f.write(content)
                f.write(self.constants_file.read())
        os.rename('constants_copy.txt', self.constants_file_path)

    def check_in_range(self, time, time_range):
        print("Checking in range ------")
        print(time)
        print(time_range)
        if time in time_range:
            self.bool = True
        else:
            self.bool = False
        print(self.bool)

    def check_for_reboot(self):
        self.get_current_time()
        self.get_five_minutes_ago()
        self.get_time_range(self.five_minutes_ago_time, self.time_now)
        self.loop_through_text_file()
        self.len_file = len(self.times_list)
        if self.len_file >= self.max_num_of_times_bf_reboot:
            for i in reversed(range(self.len_file)):
                self.extracted_time = self.times_list[i]
                self.check_in_range(self.extracted_time, self.time_range)
                if self.bool:
                    self.bool_counter += 1
                else:
                    continue
            if self.bool_counter >= self.max_num_of_times_bf_reboot:
                self.loop_through_constants_file()
                crash_email_inst = CrashReport()
                crash_email_inst.perform_job()
            if self.debug:
                print(self.bool_counter)


if __name__ == "__main__":
    anti_reboot_inst = Anti_Reboot()
    anti_reboot_inst.check_for_reboot()
