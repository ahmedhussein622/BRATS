import numpy as np
from Patient import Patient
import utility
import os

def TP (expected, reality, cdim, tdim):
    #print expected
    if cdim == tdim:
        if expected != 0 and reality != 0:
            return 1
        return 0
    ret = 0
    for i in range (0, expected.shape[0]):
        #print i
        ret = ret + TP(expected[i], reality[i], cdim+1, tdim)
    return ret

def TN (expected, reality, cdim, tdim):
    if cdim == tdim:
        if expected == 0 and reality == 0:
            return 1
        return 0
    ret = 0
    for i in range (0, expected.shape[0]):
        ret = ret + TN(expected[i], reality[i], cdim+1, tdim)
    return ret

def FP (expected, reality, cdim, tdim):
    if cdim == tdim:
        if expected != 0 and reality == 0:
            return 1
        return 0
    ret = 0
    for i in range (0, expected.shape[0]):
        ret = ret + FP(expected[i], reality[i], cdim+1, tdim)
    return ret

def FN (expected, reality, cdim, tdim):
    if cdim == tdim:
        if expected == 0 and reality != 0:
            return 1
        return 0
    ret = 0
    for i in range (0, expected.shape[0]):
        ret = ret + FN(expected[i], reality[i], cdim+1, tdim)
    return ret


def func1 (expected, reality):
    print TP (expected, reality, 0, expected.ndim)
    print TN (expected, reality, 0, expected.ndim)
    print FP (expected, reality, 0, expected.ndim)
    print FN (expected, reality, 0, expected.ndim)
    return




def patient_accuracy(patient_file):
    
    p = Patient()
    p.set_parent_file(patient_file)
    p.set_window_size(0)
    p.start_iteration()
    
    print("patient accuracy")
    print(p.file_parent)
    for z in range(0, p.limit_z):
        for y in range(0, p.limit_y):
            for x in range(0, p.limit_x):
                if(p.label[z][y][x] != 0):
                    p.label[z][y][x] = 1
    
    a = p.label
    b = utility.read_mha_image_as_nuarray(p._find_file("segmentation.mha"))
    
    print("files load , start accuracy calculation")
    
    P = TP(a, b, 0, a.ndim) + FP(a, b, 0, a.ndim) #total positive
    T = TP(a, b, 0, a.ndim) + FN(a, b, 0, a.ndim) #total actual positive
    N = TN(a, b, 0, a.ndim) + FP(a, b, 0, a.ndim) #total actual negative
    dice = TP(a, b, 0, a.ndim) / float(((P+T)/2))
    sens = TP(a, b, 0, a.ndim) / float(T)
    spec = TN(a, b, 0, a.ndim) / float(N)
    acc = (TP(a, b, 0, a.ndim) + TN(a, b, 0, a.ndim)) / float(T+N)
    error = 1.0 - acc
    
    
    p.stop_iteration()
    print("P %s" %P)
    print("T %s" %T)
    print("N %s" %N)
    print("dice %s" %dice)
    print("sens %s" %sens)
    print("spec %s" %spec)
    print("acc %s" %acc)
    print("error %s" %error)
    
    
   
    
    
file_name = "/mnt/2C88C54088C50972/school/year 4/GP/histogram_to_upload/HGG/brats_2013_pat0027_1"
patient_accuracy(file_name)
    

# a = np.array([[[1,0,1],[0,1,0],[1,1,1]],[[1,0,1],[0,1,0],[1,1,1]]])
# b = np.array([[[0,1,0],[1,0,1],[1,1,1]],[[0,1,0],[1,0,1],[1,1,1]]])

# file_name = "/mnt/2C88C54088C50972/school/year 4/GP/histogram_to_upload/HGG/brats_2013_pat0008_1"
# a_file_name = file_name+"/segmentation.mha"
# b_file_nmae = file_name+"/VSD.Brain_3more.XX.XX.OT.54559/VSD.Brain_3more.XX.XX.OT.54559.mha"
# a = utility.read_mha_image_as_nuarray(a_file_name)
# 
# b = utility.read_mha_image_as_nuarray(b_file_nmae)



# func1 (a, b)

















