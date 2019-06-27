import argparse
from timeit import default_timer as timer

import cv2
from PIL import Image
import numpy as np

from yolo.yolo import YOLO
from line.pipeline import calculate_img

def setArgument():
    parser = argparse.ArgumentParser()

    '''
    Command line options
    '''
    parser.add_argument(
        "-i","--input", nargs='?', type=str, required=False, default='test_videos/test_video.mp4',
        help="Video input path"
    )

    parser.add_argument(
        "-o","--output", nargs='?', type=str, default='',
        help="[Optional] Video output path"
    )

    flags = parser.parse_args()

    return flags

def main(flags):
    yolo = YOLO()

    vid = cv2.VideoCapture(
        flags.input) if flags.input != '' else cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    # video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC    = cv2.VideoWriter_fourcc(*'XVID')
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    isOutput = True if flags.output != '' else False

    if isOutput:
        print("!!! TYPE:", type(flags.output), type(
            video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(flags.output, video_FourCC,
                              video_fps, video_size)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while vid.isOpened():
        ret, frame = vid.read()
        if ret:
            frame=calculate_img(frame)
            image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time-prev_time
            prev_time = curr_time
            accum_time = accum_time+exec_time
            curr_fps += 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    yolo.close_session()

if __name__ == '__main__':

    flags=setArgument()
    main(flags)