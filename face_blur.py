#!/usr/bin/env python

import cv2
import numpy as np
from os.path import dirname, join
import os
import pdb
import sys
from progress.bar import ShadyBar
import glob

prototxt_path = join(dirname(__file__), "deploy.prototxt")
model_path = join(dirname(__file__), "res10_300x300_ssd_iter_140000_fp16.caffemodel")
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

class VideoTransformer():

    def __init__(self, original_dir, blurred_dir):
        self.original_dir = original_dir
        self.blurred_dir = blurred_dir
        self.prepare_frames()
        self.n_frames = self.extract_stills()
        self.bar = ShadyBar("Blurring", max = self.n_frames)
        self.codec = cv2.VideoWriter_fourcc(*'MP4V')
        self.blur_movie()
        self.bar.finish()
        print("New video is in {} with {} frames".format(self.output_video, self.n_frames))

    def prepare_frames(self):
        if len(sys.argv) > 1:
            input_video = sys.argv[1]
        else:
            input_video = "input.mp4"

        if not os.path.exists(input_video):
            print("{} does not exist".format(input_video))
            sys.exit(2)
            
        self.input_video = input_video
        self.output_video = "blurred.{}".format(input_video)

        if not os.path.exists("frames"):
            os.mkdir("frames")

        for d in (self.original_dir, self.blurred_dir):
            if not os.path.exists(d):
                os.mkdir(d)

    def extract_stills(self):
        """Assume if there are any stills, we don't need to do this."""
        existing = len(os.listdir(self.original_dir)) 
        if existing > 0:
            print("Files exist in {} : skipping still extraction from {}".format(self.original_dir, self.input_video))
            return existing
        vidcap = cv2.VideoCapture(self.input_video)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(self.original_dir, "frame%d.jpg" % count, image))
            count += 1
        return count

    def blur_movie(self):
        sample_image = cv2.imread(os.path.join(self.original_dir, "frame0.jpg"))
        height, width = sample_image.shape[:2]
        video = cv2.VideoWriter(self.output_video, self.codec, 15,  (width, height))#, True)

        for frame_id in range(0, self.n_frames):
            filename = os.path.join(self.original_dir, "frame%d.jpg" % frame_id)
            image = cv2.imread(filename)
            h, w = image.shape[:2]
            kernel_width = (w // 7) | 1
            kernel_height = (h // 7) | 1
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            model.setInput(blob)
            output = np.squeeze(model.forward())

            for i in range(0, output.shape[0]):
                confidence = output[i, 2]
                # get the confidence
                # if confidence is above 40%, then blur the bounding box (face)
                if confidence > 0.4:
                    # get the surrounding box cordinates and upscale them to original image
                    box = output[i, 3:7] * np.array([w, h, w, h])
                    start_x, start_y, end_x, end_y = box.astype(int)
                    face = image[start_y: end_y, start_x: end_x]
                    # apply gaussian blur to this face
                    face = cv2.GaussianBlur(face, (kernel_width, kernel_height), 0)
                    # put the blurred face into the original image
                    image[start_y: end_y, start_x: end_x] = face
                    video.write(image)
                    self.bar.next()
        video.release()

    def __repr__(self):
        return "A nice little Elf!"

def main():
    original_dir="./frames/original"
    blurred_dir="./frames/blurred"
    vxform = VideoTransformer(original_dir, blurred_dir)
    print(vxform)

if __name__ == '__main__':
    main()
