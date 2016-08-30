


import numpy
from Patient import Patient
import pickle

import pylab as plt






class Batch(object):
    
    def __init__(self, patients_list, window_size):
        self.patients = patients_list
        self.window_size = window_size
        
        self.n_patients = len(self.patients)
        
        for p in self.patients:
            p.set_window_size(self.window_size)
            
        self.current_patient = 0
        
        self.d = window_size / 2
        
        self.current_run = 1
        


    def get_batch(self, batch_size):
#         if(self.has_next_batch == False):
#             return None
        
        features = []
        labels = []
        
        c = 0
        p = self.patients[self.current_patient]
        
        while(c < batch_size):
            
            if(p.has_next_batch()):
                g, h = p.get_batch(batch_size - c)
                c = c + len(g)
                
                features = features + g
                labels = labels + h

            
            
            
            if(not p.has_next_batch()):
                p.stop_iteration()
                p.set_window_size(self.window_size)
                
                if(self.current_patient == self.n_patients - 1):
                    self.current_run += 1
                    
                    for pp in self.patients:
			pp.start_iteration()
                        pp.ratio = 1.0
                        pp.balance_data()
                        pp.stop_iteration()
                        
                    self.current_patient = 0
                    print("new run ! %s" %self.current_run)
                    
                else:
                    self.current_patient += 1
                    
                p = self.patients[self.current_patient]
                
                p.start_iteration()
                print ("batch moved to next patient %s" %self.current_patient)
                
            
        
        x = numpy.asarray(features)
        x = numpy.squeeze(x)
        
        y = numpy.asarray(labels)
        y = numpy.squeeze(y)
    
        return x, y




 


    def start_iteration(self):
        self.patients[self.current_patient].start_iteration()
        
    def stop_iteration(self):
        self.patients[self.current_patient].stop_iteration()
    
        
    def get_current_location(self):
        p = self.patients[self.current_patient]
        
        str0 = "run : "+str(self.current_run)
        str1 = ", patient : "+str(self.current_patient)
        str2 = ", index : "+str(p.current_voxel)
        str3 = " / "+str(p.limit_balanced)
        
        if(p.has_next_batch()):
            loc = p.balanced_data_indices[p.current_voxel]
            str4 = ", voxel : "+str(loc)
        else:
            str4 = ", no more batches "
            
        str5 = "   file : "+p.file_parent
        return str0 + str1 + str2 + str3 + str4 + str5


    
        
        
    @staticmethod
    def save_to_file(batch, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(batch, f)
        batch.fast_save(file_name)
       
    @staticmethod
    def load_from_file(file_name):
        with open(file_name, 'rb') as f:
            batch =  pickle.load(f)
            batch.fast_load(file_name)
        return batch
        
        
    def fast_save(self, file_name):
        file_name = file_name + "_extra"
        x = (self.current_run, self.current_patient, self.patients[self.current_patient].current_voxel)
        with open(file_name, 'wb') as f:
            pickle.dump(x, f)
            
    def fast_load(self, file_name):
        file_name = file_name + "_extra"
        with open(file_name, 'rb') as f:
            (self.current_run, self.current_patient, self.patients[self.current_patient].current_voxel) =  pickle.load(f)
        


# Patient.find_patients()
# b = Batch_balanced(Patient.patients_list[0:10], 17)
# b.start_iteration()
# 
# k = 10000
# (X, y) = b.get_batch(k, two_class=True)
# b.stop_iteration()
# 
# 
# print("done ")
# print(X.shape)
# print(y.shape)
# 
# for i in range(0, k):
#     plt.imshow(X[i][0], cmap = "Greys_r")
#     plt.text(0, 0, "class : "+str(y[i]), color = "r")
#     plt.show()


# 
# b.set_current_location(1, 140, 0, 0)
# 
# b.start_iteration()
# 
# 
# print(b.get_current_location())
# print(b.has_next_batch())
# x, y = b.get_batch(10000000)
# print(x.shape)


# print(b.patients[b.current_patient].get_current_input())
# print(b.patients[b.current_patient].get_current_location())
# print(b.get_current_location())
# print(b.has_next_batch())
# 
# b.stop_iteration()
























