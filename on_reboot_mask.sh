#!/bin/bash

echo "Initializing Environment..." &>/home/pi/Mask.log
# shellcheck disable=SC2046
source `which virtualenvwrapper.sh`
workon openvino
# shellcheck disable=SC1090
source ~/openvino/bin/setupvars.sh
echo "Testing Audio..." &>>/home/pi/Mask.log
aplay /home/pi/test_audio.wav
# shellcheck disable=SC2164
cd ~/Mask_Detector_v2/
# shellcheck disable=SC2129
echo "Running the Script..." &>>/home/pi/Mask.log
python3 capture_images.py &>>/home/pi/Mask.log
echo "Script Exiting..." &>>/home/pi/Mask.log
echo "Running Check_Reboot_Program" &>>/home/pi/Mask.log
python3 anti_reboot_program.py
echo "Program ran." &>>/home/pi/Mask.log
#echo "rebooting..." &>> /home/pi/Mask.log
#sudo reboot
