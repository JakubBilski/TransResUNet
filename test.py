import numpy as np
import cv2
import os
from glob import glob
from tensorflow.keras.models import load_model, Model

IMGSIZE = 128

TEST_DIR = os.path.join("..", "input", 'our_test')

# From: https://github.com/zhixuhao/unet/blob/master/data.py
def test_load_image(test_file, target_size=(256,256)):
    img = cv2.imread(test_file, cv2.IMREAD_GRAYSCALE)
    img = img / 255
    img = cv2.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img,(1,) + img.shape)
    return img

def test_generator(test_files, target_size=(256,256)):
    for test_file in test_files:
        yield test_load_image(test_file, target_size)

model = load_model(os.path.join(TEST_DIR, 'saved_model.hdf5'),
    custom_objects={'Functional':Model})
test_files = [test_file for test_file in glob(os.path.join(TEST_DIR, "*.png")) \
              if ("_mask" not in test_file \
                  and "_dilate" not in test_file \
                  and "_predict" not in test_file)]
test_gen = test_generator(test_files, target_size=(IMGSIZE,IMGSIZE))
results = model.predict_generator(test_gen, len(test_files), verbose=1)
