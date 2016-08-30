'''Train a simple convnet on the MNIST dataset.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''



from __future__ import print_function
from keras.utils import np_utils
from Batch import Batch


import model_setup
import data_setup

# to be moved to train_model
import time
import pylab as plt




def orgazize_date(X_data, y_data):
    return X_data.astype('float32'), np_utils.to_categorical(y_data, model_setup.nb_classes) 
    
#######################################################################################################3




def train(nb_epoch, n_batches, batch_size, max_runs = 1):
    
    print("+----------------------------+")
    print("|          training          |")
    print("+----------------------------+")
    
    model = model_setup.load_mode()
    training_batch = Batch.load_from_file(data_setup.file_training_batch)
    training_batch.start_iteration()
    
    
    for i in range(n_batches):
        
#         if(not training_batch.has_next_batch()):
#             print("training done, no more batches")
#             break
        print (">>>>>>>>>>>>>>>>>>>>>>>>>> training on batch : ", i)
        
        
        
        ######################################################################
        X_train, y_train =  training_batch.get_batch(batch_size)
        
        print (training_batch.get_current_location())
        
        print (X_train.shape)
        print (y_train.shape)
        
        mapper = {}
        for i in range(0, y_train.shape[0]):

            f = y_train[i]
            
            if f in mapper:
                mapper[f] = mapper[f]+1
            else:
                mapper[f] =1
        print(mapper)
        
        
        
        
        X_train, y_train = orgazize_date(X_train, y_train)
        
        
#         for i in range(0, batch_size):
#             plt.imshow(X_train[i][0], cmap = "Greys_r")
#             plt.text(0, 0, str(y_train[i]), color = 'r')
#             plt.show()
        
        
        
        
        model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
       
        
    
        ##############################################################################
        model_setup.save_model(model)
#         training_batch.stop_iteration()
#         Batch.save_to_file(training_batch, data_setup.file_training_batch)
        training_batch.fast_save(data_setup.file_training_batch)
        
        if(training_batch.current_run > max_runs):
            break
 
    model_setup.save_model(model)
    
    print ("                         +-+-+-+-+-+-+ training done +-+-+-+-+-+-+")
    
    


     






if __name__ == "__main__":

    batch_size  = 300
    n_batches = 1000000
    nb_epoch = 1
    max_runs = 1000
    
    print("start training")
    print("Batch size : %s" %batch_size)
    print("number of batches : %s" %n_batches)
    print("number of epochs : %s" %nb_epoch)
    print("maximum runs : %s" %max_runs)
    
    print("3\n2\n1\nGo !\n")

    before = int(round(time.time() * 1000))
    
    train(nb_epoch = nb_epoch, n_batches=n_batches, batch_size=batch_size, max_runs=max_runs)
    
    totoal_time = int(round(time.time() * 1000)) - before
    
    print("total time : %s minutes" %(totoal_time/1000.0/60))
  
    
    














































