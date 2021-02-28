
#!/usr/bin/env python

import cv2
import numpy as np
from os.path import dirname, join
import os
import pdb
import sys
from progress.bar import ShadyBar
import ffmpeg

prototxt_path = join(dirname(__file__), "deploy.prototxt")
model_path = join(dirname(__file__),
                  "res10_300x300_ssd_iter_140000_fp16.caffemodel")
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


class VideoTransformer():

    def __init__(self, original_dir, blurred_dir, confidence_threshold, input_video):
        self.original_dir = original_dir
        self.blurred_dir = blurred_dir
        self.confidence_threshold = confidence_threshold
        self.input_video = input_video
        self.prepare_frames()
        self.n_frames = self.extract_stills()
        self.blur_bar = ShadyBar("Blurring", max=self.n_frames)
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.blur_movie(confidence_threshold)
        self.blur_bar.finish()
        self.addAudio()
        print("New base-video is in {} with {} frames".format(
            self.output_video, self.n_frames))

    def prepare_frames(self):
        if not os.path.exists(self.input_video):
            print("{} does not exist".format(self.input_video))
            sys.exit(2)

        self.output_video = "blurred.{}".format(self.input_video)

        if not os.path.exists("frames"):
            os.mkdir("frames")

        for d in (self.original_dir, self.blurred_dir):
            if not os.path.exists(d):
                os.mkdir(d)

    def extract_stills(self):
        """Assume if there are any stills, we don't need to do this."""
        print("Extracting stills...")
        existing = len(os.listdir(self.original_dir))
        if existing > 0:
            print("Files exist in {} : skipping still extraction from {}".format(
                self.original_dir, self.input_video))
            return existing
        vidcap = cv2.VideoCapture(self.input_video)

        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            success,image = vidcap.read()
            cv2.imwrite(os.path.join(self.original_dir, "frame%d.jpg" % count), image)
            count += 1
        return count


    def blur_movie(self, confidence_threshold):
        
        # For the FPS
        vidcap = cv2.VideoCapture(self.input_video)
        
        sample_image = cv2.imread(os.path.join(
            self.original_dir, "frame0.jpg"))
        height, width = sample_image.shape[:2]
        video = cv2.VideoWriter(
            self.output_video, self.codec, vidcap.get(cv2.CAP_PROP_FPS),  (width, height))  # , True)

        for frame_id in range(0, self.n_frames):
            filename = os.path.join(
                self.original_dir, "frame%d.jpg" % frame_id)
            image = cv2.imread(filename)
            h, w = image.shape[:2]
            #print("Processing {} {}x{}".format(filename, h,w))
            kernel_width = (w // 7) | 1
            kernel_height = (h // 7) | 1
            blob = cv2.dnn.blobFromImage(
                image, 1.0, (300, 300), (104.0, 177.0, 123.0))
            model.setInput(blob)
            output = np.squeeze(model.forward())

            for i in range(0, output.shape[0]):
                confidence = output[i, 2]
                # get the confidence
                # if confidence is above 40%, then blur the bounding box (face)
                if confidence > confidence_threshold:
                    # get the surrounding box cordinates and upscale them to original image
                    box = output[i, 3:7] * np.array([w, h, w, h])
                    start_x, start_y, end_x, end_y = box.astype(int)
                    face = image[start_y: end_y, start_x: end_x]
                    # apply gaussian blur to this face
                    face = cv2.GaussianBlur(
                        face, (kernel_width, kernel_height), 0)
                    # put the blurred face into the original image
                    image[start_y: end_y, start_x: end_x] = face
            video.write(image)
            self.blur_bar.next()
        video.release()

    def addAudio(self):
        stream_input = ffmpeg.input(self.input_video)
        stream_output = ffmpeg.input(self.output_video)
        audio = stream_input.audio
        video = stream_output.video
        out = ffmpeg.output(audio, video, 'blurred_with_audio.mp4')
        print("Adding audio...")
        ffmpeg.run(out, capture_stdout=False, capture_stderr=True, overwrite_output=True)
        print("New audio-video is in blurred_with_audio.mp4")

    def __repr__(self):
        return "A nicer little Elf!"


def main():
    original_dir = "./frames/original"
    blurred_dir = "./frames/blurred"
    if len(sys.argv) > 1:
        confidence_threshold = float(sys.argv[1])
    else:
        confidence_threshold = 0.4
    if len(sys.argv) > 2:
        input_video = sys.argv[2]
    else:
        input_video = "input.mp4"

    
    
    vxform = VideoTransformer(
        original_dir, blurred_dir, confidence_threshold, input_video)
    print(vxform)


if __name__ == '__main__':
    main()
