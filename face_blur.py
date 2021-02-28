#!/usr/bin/env python

import cv2
import numpy as np
from os.path import dirname, join
import os
import pdb
import sys
from progress.bar import ShadyBar
import ffmpeg
import argparse
import logging

prototxt_path = join(dirname(__file__), "deploy.prototxt")
model_path = join(dirname(__file__),
                  "res10_300x300_ssd_iter_140000_fp16.caffemodel")
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

_DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.DEBUG)

_LOGGER = logging.getLogger(__name__)

_LOGGER.info("Model loaded")

from art import ascii_dog

class VideoTransformer():

    def __init__(self, original_dir, args):
        self.original_dir = original_dir
        self.confidence_threshold = args.confidence
        self.input_video = args.input
        self.output_video = "blurred.{}".format(self.input_video)
        self.prepare_frames()
        self.n_frames = self.extract_stills()
        self.blur_bar = ShadyBar("Blurring", max=self.n_frames)
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.blur_movie(self.confidence_threshold)
        self.blur_bar.finish()
        self.addAudio()
        _LOGGER.info("New base-video is in {} with {} frames".format(
            self.output_video, self.n_frames))

    def prepare_frames(self):
        if not os.path.exists(self.input_video):
            _LOGGER.info("{} does not exist".format(self.input_video))
            sys.exit(2)

        if not os.path.exists("frames"):
            os.mkdir("frames")
        
        if not os.path.exists("frames/original"):
            os.mkdir("frames/original")

    
    def extract_stills(self):
        """Assume if there are any stills, we don't need to do this."""
        _LOGGER.info("Extracting stills...")
        existing = len(os.listdir(self.original_dir))
        if existing > 0:
            _LOGGER.info("Files exist in {} : skipping still extraction from {}".format(
                self.original_dir, self.input_video))
            return existing
        vidcap = cv2.VideoCapture(self.input_video)

        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            try:
                success,image = vidcap.read()
                cv2.imwrite(os.path.join(self.original_dir, "frame%d.jpg" % count), image)
                count += 1
            except Exception:
                pass
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
        _LOGGER.info("Adding audio...")
        ffmpeg.run(out, capture_stdout=False, capture_stderr=True, overwrite_output=True)
        _LOGGER.info("New audio-video is in blurred_with_audio.mp4")

    def __repr__(self):
        r = []
        r.append("VideoEncoder class brought you via")
        r.append("A nicer little Elf!")
        r.append("With help from Torin Stephen, Trixie Twizzle and Sun An Tian")
        r.append(ascii_dog)
        return '\n'.join(r)
        


def main():
    p = argparse.ArgumentParser(description='Torin Auto Blur Face tool')

    #p.add_argument('-i', '--input', dest='inputt', help='Video to process.', default='input.mp4')
    p.add_argument('-i', '--input',  help='Video to process.', default='input.mp4')
    p.add_argument('-o', '--output',  help='Video result.', default='output.mp4')
    p.add_argument('-c', '--confidence',  type=float, help='Confidence threshold to blur pixels 0.0 - 1.0', default=0.4)
    p.add_argument('-n', '--no-audio',  action='store_true', help='Exclude audio track')
    p.add_argument('-C', '--codec',  type=str, choices=['mpv4','trixie'], help='CODEC choice', default='mpv4')
    p.add_argument('-f', '--frames',  type=str,  help='Number of frames to encode', default=200)
    
#    p.add_argument('-u', '--user', type=str, metavar='USER', help='Github username.', required=True)
#    p.add_argument('-v', '--version', type=parse_version, metavar='VERSION', help='Use given version for the release. If version contains only `+\' signs then it will increase latest version number: one `+\' increases major version number (e.g. 1.2.3 -> 2.0), `++\' increases minor version number (e.g. 1.2.3 -> 1.3), `+++\' increases patch level (e.g. 1.2.3 -> 1.2.4). Defaults to `+++\'.', default='+++')
#    p.add_argument('-r', '--rev', metavar='COMMIT', help='Use given revision for the release. Defaults to `develop\'.', default='develop')
#    p.add_argument('-s', '--stages', action='append', metavar='STAGE', help='Only run one of the given stages (default to all).', choices=tuple((stname for stname, stfunc in stages)))
#    p.add_argument('-p', '--password', type=str, metavar='PASS', help='Github password. You will be prompted if it is not supplied.')
#    p.add_argument('-o', '--overlay', type=str, metavar='PATH', help='Location of the local clone of the {0} overlay. If provided directory does not exist it will be created by “git clone”. Defaults to /tmp/powerline-{0}.'.format(OVERLAY_NAME), default='/tmp/powerline-' + OVERLAY_NAME)
#
    args = p.parse_args()

    if args.confidence > 1.0:
        args.confidence = 1.0
    elif args.confidence < 0.0:
        args.confidence = 0.0

    confidence_threshold = args.confidence

    original_dir = "./frames/original"
    input_video = args.input
    _LOGGER.info("input video is {}".format(input_video))
    _LOGGER.info("{}".format(args))
            
    
    vxform = VideoTransformer( original_dir, args)
    _LOGGER.info(vxform)


if __name__ == '__main__':
    main()
