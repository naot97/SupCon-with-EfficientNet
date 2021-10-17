#from download import download_clip
from os import path
import os
from os import listdir
from os.path import isfile, join
import cv2

datadir = 'test_video/apex/'
videos = [f for f in listdir(datadir) if f[-3:] == 'mp4']
for video in videos:
    try:
        name = video.split('_')
        if 'mp4' not in name[1]:
            os.rename(datadir +  video, datadir + name[0] + '_' + name[1] + '.mp4')
    except:
        pass




all_slugs = []
videos = [f for f in listdir(datadir) if f[-3:] == 'mp4']
for video_file in videos:

    cap = cv2.VideoCapture(datadir + video_file)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = int(frame_count/fps)
        for i in range(0, duration, 1):
            cap.set(cv2.CAP_PROP_POS_MSEC, i * 1)
            ret, frame = cap.read()
            basename, ext = path.splitext(video_file)
            target_img = path.join(datadir, '{}_{}.jpg'.format(basename, i))
            cv2.imwrite(target_img, frame)
    finally:
        cap.release()
