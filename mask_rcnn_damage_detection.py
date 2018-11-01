import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import base64

# Import Mask RCNN
import mrcnn.model as modellib
from mrcnn.config import Config

CLASS = ['BG', 'damage']

# Root directory of the project
ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

WEIGHTS_PATH = 'mask_rcnn_damage_0010.h5'

class CustomConfig(Config):
	"""Configuration for training on the toy  dataset.
	Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name
	NAME = "damage"

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # BG + damage

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 100

	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.90
	
	# Batch size is GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1	
	IMAGES_PER_GPU = 1

config = CustomConfig()
config.display()

DEVICE = '/cpu:0' # '/gpu:0'

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode = 'inference', model_dir = MODEL_DIR, config = config)
	
# Load weights
print('Loading weights ', WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, by_name = True)

# class id is 1, ['link']

def detect(img, verbose = False):
	ret = []
	try:
		img = base64.b64decode(img)
		img = np.frombuffer(img, np.uint8)
		img = cv2.imdecode(img, -1)
		r 	= model.detect([img], verbose = verbose)[0]
		for index, zipped_bbox_class_id in enumerate(zip(r['rois'], r['class_ids'])):
			bbox, class_id = zipped_bbox_class_id
			if class_id >= 1:
				coords_of_class = []
				y1, x1, y2, x2 	= bbox
				for row in range(y1, y2 + 1):
					for col in range(x1, x2 + 1):
						if r['masks'][row][col][index]:
							pt = row, col
							coords_of_class.append(pt)
				ret.append((coords_of_class, CLASS[class_id]))#, LINK[class_id]))
	except:
		pass
	return ret