#!/usr/bin/env python

import cv2
import numpy as np
from os.path import dirname, join
import os
import pdb
import sys
from progress.bar import ShadyBar

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
prototxt_path = join(dirname(__file__), "deploy.prototxt")
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
model_path = join(dirname(__file__),
                  "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# load Caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

if len(sys.argv) > 1:
    input_video = sys.argv[1]
else:
    input_video = "input.mp4"

if not os.path.exists(input_video):
    print("{} does not exist".format(input_video))
    sys.exit(2)

output_video = "blurred.{}".format(input_video)

original_dir="./frames/original"
blurred_dir="./frames/blurred"

if not os.path.exists("frames"):
    os.mkdir("frames")

for d in (original_dir, blurred_dir):
    if not os.path.exists(d):
        os.mkdir(d)

def extract_stills(video, target_directory):
    """Assume if there are any stills, we don't need to do this."""
    count = 0
    if len(os.listdir(target_directory)) > 0:
        print("Files exist in {} : skipping still extraction from {}".format(target_directory, video))
        return count
    vidcap = cv2.VideoCapture(video)
    success, image = vidcap.read()
    while success:
        # replace by original_dir
        cv2.imwrite("./frames/original/frame%d.jpg" %
                    count, image)     # save frame as JPEG file
        success, image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1
    return count

count = extract_stills(input_video, original_dir)
print("Extracted {} stills from {}".format(count, input_video))

count = 0
#pdb.set_trace()
images = os.listdir(original_dir)
n_frames = len(images)
sample_image = cv2.imread(os.path.join(original_dir, images[0]))
height, width = sample_image.shape[:2]
codec = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
codec = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter(output_video, codec, 20,  (width, height), True)
bar = ShadyBar("Blurring", max = n_frames)

for filename in os.listdir(original_dir):
    full_path = os.path.join(original_dir, filename)
    image = cv2.imread(full_path)
    #    print(image)
    # get width and height of the image
    h, w = image.shape[:2]
    # gaussian blur kernel size depends on width and height of original image
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    # preprocess the image: resize and performs mean subtraction
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    # set the image into the input of the neural network
    model.setInput(blob)
    # perform inference and get the result
    output = np.squeeze(model.forward())

    for i in range(0, output.shape[0]):
        confidence = output[i, 2]
        # get the confidence
        # if confidence is above 40%, then blur the bounding box (face)
        if confidence > 0.4:
            # get the surrounding box cordinates and upscale them to original image
            box = output[i, 3:7] * np.array([w, h, w, h])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(int)
            # get the face image
            face = image[start_y: end_y, start_x: end_x]
            # apply gaussian blur to this face
            face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
            # put the blurred face into the original image
            image[start_y: end_y, start_x: end_x] = face
            #blurred = os.path.join(blurred_dir, "blurred-face{}.jpg".format(count))
            #cv2.imwrite(blurred, image)
            video.write(image)
            count += 1
            bar.next()

#cv2.destroyAllWindows()
video.release()
bar.finish()
print("New video is in {} with {} frames".format(output_video, count))
