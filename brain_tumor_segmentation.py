'''Train a simple convnet on the MNIST dataset.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''


from __future__ import print_function
from keras.utils import np_utils
from Batch import Batch

import time
import model_setup
import data_setup

import numpy
from Patient import Patient
import pickle
# from bzrlib.groupcompress import BATCH_SIZE
import utility
import os

# to be moved to train_model
import threading
# from Cython.Compiler.Main import verbose

from multiprocessing import Process, Value
import re


def orgazize_data(X_data, y_data):
    return X_data.astype('float32')/255, np_utils.to_categorical(y_data, model_setup.nb_classes) 
    
#######################################################################################################3

    
   
    
def segment_unparalel(patient_parent_file, description, model, extra_file):
    
    before = int(round(time.time() * 1000))
    print("unparalel segmentation")
    print(patient_parent_file)
    p = Patient()
    p.set_parent_file(patient_parent_file)
    p.set_window_size(model_setup.window_size)
 
    p.start_iteration()
    
    print(p.file_FLAIR)
    image_name =  re.findall(r'\d+', p.file_FLAIR)
    image_name = image_name[len(image_name) - 1]
    
    path_ttt, file_nnn = os.path.split(patient_parent_file)
    image_name = "VSD."+description+"_("+file_nnn+")."+image_name+".mha"
    print("result name : %s" %image_name)
     
    file_result_name = os.path.join(patient_parent_file, image_name)
    extra_file = os.path.join(extra_file, image_name)
    
    if(os.path.isfile(file_result_name)):
        return
    
    segmentation = numpy.zeros(shape = p.label.shape)
    z = 0
    utility.save_nuarray_as_mha(file_result_name, segmentation)
    utility.save_nuarray_as_mha(extra_file, segmentation)

    
   
    while z < p.limit_z:
        for y in range(0, p.limit_y):
            if(y % 10 == 0):
                print("%s %s" %(z, y))
            for x in range(0, p.limit_x):
                if(p.is_back_ground(z, y, x)):
                    segmentation[z][y][x] = 0
                else :
                    t, g = p.get_batch_at(z, y, x)
                    features = []
                    features.append(t)
                    features = numpy.asarray(features)
                    features = features.astype('float32')
                    r = model.predict_classes(features, batch_size=1, verbose = False)
                    segmentation[z][y][x] = r

        z += 1
        


    utility.save_nuarray_as_mha(file_result_name, segmentation)
    utility.save_nuarray_as_mha(extra_file, segmentation)
    p.stop_iteration()
    print("segmentation done")
    
    totoal_time = int(round(time.time() * 1000)) - before
    print("total time : %s minutes" %(totoal_time/1000.0/60))
    
    

if __name__ == "__main__":
 
#     thread_count = 8

#     file_name = "/mnt/2C88C54088C50972/school/year4/GP/histogram_to_upload/HGG/brats_2013_pat0013_1"
#     file_name = "/mnt/01D11BE6ACC16AC0/GP/Preprocessed_Training/HGG/brats_2013_pat0001_1"

    
    
#     before = int(round(time.time() * 1000))
    
#     segment(file_name, thread_count)
#     segment_slice(file_name, 77, file_name+"/slice_segmentation.mha")
#     segment_unparalel(file_name)

    
#     totoal_time = int(round(time.time() * 1000)) - before
#     print("total time : %s minutes" %(totoal_time/1000.0/60))

    utility.BRATS_2015_parent_file = "/media/Data/Ahmed/Preprocessed_Testing"
    extra_file = "/media/Data/Ahmed/testing_result"
    Patient.find_patients()
    
    model = model_setup.load_mode()
    
    for p in Patient.patients_list:
        segment_unparalel(p.file_parent, "CNN_69X69", model, extra_file)
   
    
    
    
















































