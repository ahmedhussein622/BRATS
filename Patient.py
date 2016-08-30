import os
import numpy
import  re, gc
from collections import deque
import utility
from copy_reg import pickle
import pickle
import random
import pylab as plt


class Patient(object):
    
    
    def __init__(self) :
        self.file_parent = None
        self.file_T1 = None
        self.file_T1c = None
        self.file_T2 = None
        self.file_FLAIR = None
        self.file_label = None
        
        self.T1 = None
        self.T1c = None
        self.FLAIR = None
        self.T2 = None
        self.label = None
        
        self.ratio = 1.0
        
       
        
    def set_window_size(self, window_size):
        self.window_size = window_size
        self.d = window_size / 2    
        self.current_voxel = 0
        


    def start_iteration(self):    
        self.load_patient_files()
        self.preprocess()
        
        shape = self.label.shape
        self.limit_z = shape[0]
        self.limit_y = shape[1]
        self.limit_x = shape[2]
        
        self.limit_balanced = len(self.balanced_data_indices)
            
#         d = int(self.window_size / 2)
        npad = ((0,0), (self.d,self.d), (self.d,self.d))
        self.T1 = numpy.pad(self.T1, pad_width = npad, mode = 'constant', constant_values = 0)
        self.T1c = numpy.pad(self.T1c, pad_width = npad, mode = 'constant', constant_values = 0)
        self.T2 = numpy.pad(self.T2, pad_width = npad, mode = 'constant', constant_values = 0)
        self.FLAIR = numpy.pad(self.FLAIR, pad_width = npad, mode = 'constant', constant_values = 0)
        
         
        
    
    def stop_iteration(self):
        self.unload_patient_files()
        self.postprocess()   
        
#######################################################################################

    def balance_data(self):
        print("balancing data")
        maper = {}
        tumor = []
        healty = []
        for z in range(0, self.label.shape[0]):
            for y in range(0, self.label.shape[1]):
                for x in range(0, self.label.shape[2]):
                    if(self.back_ground[z][y][x] == 0):
                        f = self.label[z][y][x]
                        if(f in maper):
                            maper[f] = maper[f] + 1
                        else:
                            maper[f] = 1
                            
                        if(f != 0):
                            tumor.append((z, y, x))
                        else:
                            healty.append((z, y, x))   
                            
        print(maper)
        random.shuffle(tumor)
        random.shuffle(healty)        
        
        
        kk = min(int(len(tumor) * self.ratio), len(healty))
        all_tissue = tumor + healty[0 : kk]
        
        
        random.shuffle(all_tissue)
#             all_tissue = all_tissue[0 : len(all_tissue) / 2] #becafeful
        
	print(self.ratio)
        print(len(tumor))
        print(kk)
        print(len(all_tissue))
        
    
        balanced_data_file = os.path.join(self.file_parent, "balanced_data.pickle")
        ff = open(balanced_data_file, "wb")
        pickle.dump(all_tissue, ff)
        ff.close()

    def preprocess(self):
        self.set_parent_file(self.file_parent)
        
        if(self._find_file("back_ground") == None):
            back_ground = self.BFS()
            utility.save_nuarray_as_mha(os.path.join(self.file_parent, "back_ground.mha"), back_ground)
            self.set_parent_file(self.file_parent)
        
        self.back_ground = utility.read_mha_image_as_nuarray(self._find_file("back_ground\.mha"))
        
        balanced_data_file = os.path.join(self.file_parent, "balanced_data.pickle")
        if(not os.path.isfile(balanced_data_file)):
            self.balance_data()
            
            
        ff = open(balanced_data_file, "rb")
        self.balanced_data_indices = pickle.load(ff)
        ff.close()
         


    def postprocess(self):
        self.back_ground = None    
        
        
        
    def BFS(self):
        print("BFS")
        print(self.file_parent)
        dx = [1, -1, 0, 0, 0, 0]
        dy = [0, 0, 1, -1, 0, 0]
        dz = [0, 0, 0, 0, 1, -1]
        
        visited = numpy.zeros(shape = self.T1.shape, dtype = numpy.int8)
        visited[0][0][0] = 1
        queue = deque()
        queue.append((0, 0, 0))
        
        
        s = self.T1.shape
        limit_z = s[0]
        limit_y = s[1]
        limit_x = s[2]
#         c = 0
        while queue:
            node = queue.popleft()
            visited[node[0], node[1], node[2]] = 1
            for k in range(len(dx)):
                
                z_new = node[0] + dz[k]
                y_new = node[1] + dy[k]
                x_new = node[2] + dx[k]
                
                if((x_new >= 0 and x_new < limit_x) and (y_new >= 0 and y_new < limit_y)
                    and (z_new >= 0 and z_new < limit_z) and (visited[z_new][y_new][x_new] == 0)):
                    
                    if(self.T1[z_new][y_new][x_new] == 0 and self.T1c[z_new][y_new][x_new] == 0 and
                       self.T2[z_new][y_new][x_new] == 0 and self.FLAIR[z_new][y_new][x_new] == 0):
                        
                        visited[z_new][y_new][x_new] = 1
                        queue.append((z_new, y_new, x_new))
        
        return visited   
        
    def get_batch_at(self, z, y, x):
        
        
        d = self.d
        
        x_from = x
        x_to = x +  2 * d + 1
        y_from = y
        y_to = y + 2 * d + 1  
    
        result = []
        result.append(numpy.squeeze(self.T1[z : z+1, y_from: y_to, x_from: x_to]))
        result.append(numpy.squeeze(self.T1c[z : z+1, y_from: y_to, x_from: x_to]))
        result.append(numpy.squeeze(self.T2[z : z+1, y_from: y_to, x_from: x_to]))
        result.append(numpy.squeeze(self.FLAIR[z : z+1, y_from: y_to, x_from: x_to]))
        label = numpy.squeeze(self.label[z : z+1, y: y + 1, x : x + 1])
        
        
        result = numpy.asarray(result)
        return result, label
    
    def is_back_ground(self, z, y, x):
        return (self.back_ground[z][y][x] == 1)
      
########################################################################################
    patients_list = []
    patients_HGG = []
    patients_LGG = []
    
    
    @staticmethod
    def find_patients():
        
        print ("searching patients")
        HGG_file = os.path.join(utility.BRATS_2015_parent_file,"HGG")
        LGG_file = os.path.join(utility.BRATS_2015_parent_file,"LGG")
        
        
        HGG_patient_files=[os.path.join(HGG_file, name) for name in os.listdir(HGG_file) if os.path.isdir(os.path.join(HGG_file, name))]
        LGG_patient_files=[os.path.join(LGG_file, name) for name in os.listdir(LGG_file) if os.path.isdir(os.path.join(LGG_file, name))]
        
        
        patient_id = 0
        for s in HGG_patient_files :
            p = Patient()
            p.patient_id = patient_id
            patient_id += 1
            p.HGG = True
            p.set_parent_file(s)
            Patient.patients_HGG.append(p)
            
        
        for s in LGG_patient_files :
            p = Patient()
            p.patient_id = patient_id
            patient_id += 1
            p.LGG = True
            p.set_parent_file(s)
            Patient.patients_LGG.append(p)
            
        Patient.patients_list = Patient.patients_HGG + Patient.patients_LGG
        Patient.n_patients = len(Patient.patients_list)
        print "patients found"     
            
    
     
    
        
    def set_parent_file(self, parent_file):
        
        self._files_names = []

        self.file_parent = parent_file
        for dirName, subdirList, fileList in os.walk(self.file_parent):
            for fname in fileList: 
                self._files_names.append(os.path.join(dirName,fname))
                
       
                        
    def _find_file(self, regex):    
        for i in range(0, len(self._files_names)):
            match = re.search(regex, self._files_names[i])
            if match != None:
                return os.path.join(self._files_names[i])
                
        return None
            
            
    def load_patient_files(self):
       

	self.file_T1 = self._find_file("T1.*\.mha")
        self.file_T1c = self._find_file("T1c.*\.mha")
        self.file_T2 = self._find_file("T2.*\.mha")
        self.file_FLAIR = self._find_file("Flair.*\.mha")

        self.T1 = utility.read_mha_image_as_nuarray(self._find_file("T1.*\.mha"))
        self.T1c = utility.read_mha_image_as_nuarray(self._find_file("T1c.*\.mha"))
        self.T2 = utility.read_mha_image_as_nuarray(self._find_file("T2.*\.mha"))
        self.FLAIR = utility.read_mha_image_as_nuarray(self._find_file("Flair.*\.mha"))
        self.label = utility.read_mha_image_as_nuarray(self._find_file("OT.*\.mha"))      
            

        
    def unload_patient_files(self):
        self.T1 = None
        self.T1c = None
        self.T2 = None
        self.FLAIR = None
        self.label = None 
              
        gc.collect()
#########################################################################################

    def has_next_batch(self):
        return self.current_voxel < self.limit_balanced
    
    
    def get_batch(self, batch_size):
        if(not self.has_next_batch()):
            return None
        
    
        features = []
        labels = []
        
        g = 0
        while(g < batch_size and self.has_next_batch()):
            
            
            (cz, cy, cx) = self.balanced_data_indices[self.current_voxel]
            
            (Xs, Ys) = self.get_batch_at(cz, cy, cx)
#             utility.show_image(Xs[0], str(cx)+", "+str(cy)+" , "+str(cx))
            
            features.append(Xs)
            labels.append(Ys)
            
            g = g + 1
            self.current_voxel = self.current_voxel + 1
            
                 
        
#         for i in range(0, len(labels)):
#             if(labels[i] != 0):
#                 labels[i] = 1
                            
        return features, labels
  
  
######################################################################################### 
  
  
  
# Patient.find_patients()
# for p in Patient.patients_list :
#     print p.file_parent
  
  
  
  
# p = Patient.patients_list[0]
# 
# p.set_window_size(70)
# p.start_iteration()
# 
# X, y = p.get_batch(50)
# 
# for t in X:
#     utility.show_image(t[0])
# 
# 
# p.stop_iteration()
# print("done")
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
        
        
        
        
        
        
        
        
        
        
