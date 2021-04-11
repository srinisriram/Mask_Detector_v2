RUN_PROGRAM = True

prototxt_path = "Production/face_detection_model/deploy.prototxt"

model_path = "Production/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

mask_model_path = "April_2nd_detector.onnx"

sound_file = "mask_warning_1.wav"

MIN_CONFIDENCE = 0.5

MIN_CONFIDENCE_DATASET = 0.995

frame_width_in_pixels = 320

OPEN_DISPLAY = True

USE_VIDEO = True

VID_CAM_INDEX = 0

MODEL_INPUT_SIZE = 128

MASK_INDEX = 0

NO_MASK_INDEX = 1

SLEEP_TIME_AMOUNT = 2

face_image_path = "face_images/"

full_frame_path = "full_frame_images/"

mask_full_folder = 'full_frame_images/with_mask/'

without_mask_full_folder = 'full_frame_images/without_mask/'

mask_face_folder = 'face_images/with_mask/'

without_mask_face_folder = 'face_images/without_mask/'

ALL_IMAGES_PATH = "Trial_DIR_Before/without_mask/*.png"

SAVE_IMAGES_PATH = "Trial_DIR_After/without_mask/"

FULL_TEMPLE_DATASET = "Temple_Dataset/"

CONSTANT_IMAGE_PATH = "face_images"

FINAL_DATASET = 'temple_final_dataset/'

MIN_THRESHOLD = 200

SR_MODEL_PATH = 'Production/ESPCN_x2.pb'

NEW_DATA_PATH = 'temple_final_dataset/'

TRASH_IMAGES = 'Trash_images/'

LABELS = ["With Mask", "Without Mask"]

COLORS = [(0, 255, 0), (0, 0, 255)]

CRASH_LOG = 'crash_log.txt'

MAX_NUM_OF_TIMES_BF_REBOOT = 5

CONSTANTS_FILE = "Production/constants.py"

LOG_FILE = '/home/pi/Mask.log'
