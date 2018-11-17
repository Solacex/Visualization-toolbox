import os
import sys
from multiprocessing import Pool, current_process
import argparse
import threading
import imageio
import skimage
import pylab
import numpy as np
from PIL import Image
import cv2
ori_path = sys.argv[1]
out_path = sys.argv[2]

NUM_THREADS = 100
def split(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def dump_frames(vid_path):
    #import cv2
    vid = imageio.get_reader(vid_path, 'ffmpeg')
    vid_name = vid_path.split('/')[-1].split('.')[0]
    out_full_path = os.path.join(out_path, vid_name)

    #fcount = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass
    file_list = []
    for num, im in enumerate(vid):
        image = skimage.img_as_float(im).astype(np.float64)

        cv2.imwrite('{}/{:06d}.jpg'.format(out_full_path, num), image)
        #access_path = '{}/{:06d}.jpg'.format(vid_name, i)
       # file_list.append(access_path)
    print('{} done'.format(vid_name), 'totoal frames:', num)

def target(video_list):
    for video in video_list:
#        os.makedirs(os.path.join(FRAME_ROOT, video[:-5]))
        dump_frames(video)


print(ori_path)
print(out_path)
video_path = []
for root, dirs, files in os.walk(ori_path):
    for d in dirs:
        for rr, dd, ff in os.walk(os.path.join(root,d)):
            for fff in ff:
                tmp_path = os.path.join(rr, fff)
                #print('starting video:', tmp_path)
                video_path.append(tmp_path)

target(video_path)
