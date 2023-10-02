#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import skimage.io as io
import skimage.transform as trans

import os
import sys
import logging as log
import numpy as np
import h5py
import time 
#import tensorflow as tf
from inference import Network
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))
from argparser import args
import cv2
from PIL import Image
def print_stats(exec_net, input_data, n_channels, batch_size, input_blob, out_blob, args):
    """
    Prints layer by layer inference times.
    Good for profiling which ops are most costly in your model.
    """

    # Start sync inference
    print("Starting inference ({} iterations)".format(args.number_iter))
    infer_time = []

    for i in range(args.number_iter):
        input_data_transposed_1=input_data[0:batch_size].transpose(0,3,1,2)
        t0 = time.time()
        res = exec_net.infer(inputs={input_blob: input_data_transposed_1[:,:n_channels]})
        infer_time.append((time.time() - t0) * 1000)


    average_inference = np.average(np.asarray(infer_time))
    print("Average running time of one batch: {:.5f} ms".format(average_inference))
    print("Images per second = {:.3f}".format(batch_size * 1000.0 / average_inference))

    perf_counts = exec_net.requests[0].get_perf_counts()
    log.info("Performance counters:")
    log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format("name",
                                                         "layer_type",
                                                         "exec_type",
                                                         "status",
                                                         "real_time, us"))
    for layer, stats in perf_counts.items():
        log.info("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                             stats["layer_type"],
                                                             stats["exec_type"],
                                                             stats["status"],
                                                             stats["real_time"]))

def load_data():
    """
    Modify this to load your data and labels
    """

    # Load data
    # You can create this Numpy datafile by running the create_validation_sample.py script
    df = h5py.File(data_fn, "r")
    imgs_validation = df["imgs_validation"]
    msks_validation = df["msks_validation"]
    img_indicies = range(len(imgs_validation))

    """
    OpenVINO uses channels first tensors (NCHW).
    TensorFlow usually does channels last (NHWC).
    So we need to transpose the axes.
    """
    input_data = imgs_validation
    msks_data = msks_validation
    return input_data, msks_data, img_indicies


def calc_dice(y_true, y_pred, smooth=1.):
    """
    Sorensen Dice coefficient
    """
    numerator = 2.0 * np.sum(y_true * y_pred) + smooth
    denominator = np.sum(y_true) + np.sum(y_pred) + smooth
    coef = numerator / denominator

    return coef

def plotDiceScore(pred_mask,plot_result):

    if plot_result:
        plt.figure(figsize=(15, 15))
    
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask[0,0, :, :], origin="lower")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# Create output directory for images
png_directory = args.results_directory
if not os.path.exists(png_directory):
    os.makedirs(png_directory)

data_fn = args.data_file
if not os.path.exists(data_fn):
    print("Wrong input path or File not exists")
    sys.exit(1)


log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

model_xml = args.model
TARGET_DEVICE = args.device
CPU_EXTENSION = args.cpu_extension
print(model_xml,TARGET_DEVICE,CPU_EXTENSION)
Net = Network()
log.info("Loading U-Net model")
[batch_size, n_channels, height, width], exec_net, input_blob, out_blob = Net.load_model(model_xml, TARGET_DEVICE, 1, 1, 0, CPU_EXTENSION)[1:5]

frame = cv2.imread('test_img/0.png',0)
prepimg = frame[:, :].copy()

prepimg = Image.fromarray(prepimg)
prepimg = prepimg.resize((256, 256), Image.ANTIALIAS)
prepimg = np.asarray(prepimg) / 255.0
prepimg=prepimg.reshape((1, 256, 256, 1))

prepimg = prepimg.transpose((0, 3, 1, 2))
start_time = time.time()
res = exec_net.infer(inputs={input_blob:prepimg})
predictions = res[out_blob]

time_elapsed = time.time()-start_time
FPS=np.around(1/time_elapsed,2)
print('FPS:',FPS)
outputs = predictions.transpose((2, 3, 1, 0)).reshape((256, 256)) # (256, 256 3)
outputs = np.reshape(outputs, (256, 256))*255.0
image = Image.fromarray(np.uint8(outputs), mode="P")

image = np.asarray(image)
print('=============',np.min(image),',',np.max(image))
cv2.imshow("frame", frame)
cv2.imshow("Result", image)
cv2.waitKey(0)


Net.clean()
