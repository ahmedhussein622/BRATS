

from Patient import Patient
from Batch import Batch
import model_setup




file_training_batch = "training_batch"
file_testing_batch = "testing_batch"



def setup_files():
    
    print ("======> warning : you called setup_files from data_setup, all previous batches are gone <======")
          
    window_size = model_setup.window_size
    
    Patient.find_patients()
    training_batch = Batch(Patient.patients_list[:201], window_size)
#     testing_batch = Batch(Patient.patients_list[201:], window_size)
 
    
    training_batch.start_iteration()
    training_batch.stop_iteration()
    Batch.save_to_file(training_batch, file_training_batch)
#     Batch.save_to_file(testing_batch, file_testing_batch)
    
   
        
    print ("============> state setup done <============")




if __name__ == "__main__":
    setup_files()













