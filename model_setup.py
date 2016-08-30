'''
    run this script to setup the files and start over.
    be careful or every thing will be lost
'''



from __future__ import print_function


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import model_from_json
import keras

file_model = "model_definition"
file_param = "model_weights.h5"


window_size = 69 #remove me
# window_size = 19
nb_classes = 5






def save_model(model):
    model.save_weights(file_param, overwrite=True)
    print("model parameters saved")
    
def load_mode():
#     model = model_from_json(open(file_model).read())
    model = compile_model()
    model.load_weights(file_param)
    print("model loaded")
    return model


def big_model():
    
    
    model = Sequential()
     
    model.add(Convolution2D(32, 3, 3,border_mode='valid', input_shape=(4, window_size, window_size), init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
    model.add(Convolution2D(32, 3, 3, init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print(model.output_shape)
     
    #########################################
     
    model.add(Convolution2D(32, 3, 3, init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
    model.add(Convolution2D(64, 3, 3, init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print(model.output_shape)
     
    #########################################
     
    model.add(Convolution2D(64, 3, 3, init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
    model.add(Convolution2D(64, 3, 3, init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print(model.output_shape)
     
    #########################################
     
    model.add(Convolution2D(96, 3, 3, init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
    model.add(Convolution2D(96, 3, 3, init='uniform'))
    model.add(Activation('relu'))
    print(model.output_shape)
     
     
    ############################################
     
    model.add(Dropout(0.5))
     
     
    #########################################
     
    model.add(Convolution2D(512, 1, 1))
    model.add(Activation('relu'))
    print(model.output_shape)
#      
#     model.add(Convolution2D(5, 1, 1))
#     model.add(Activation('relu'))
     
     
    ############################################
     
     
    model.add(Flatten())
    
#     model.add(Dense(512, init='uniform'))
#     model.add(Activation('relu'))
#     
#     model.add(Dense(64, init='uniform'))
#     model.add(Activation('relu'))
#        
    
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    print(model.output_shape)

     
    
    return model
    
    
def small_model():   
    
   
    
    #     #model :
    model = Sequential()
       
    # #layer 1 :
    model.add(Convolution2D(64, 5, 5, border_mode='valid', input_shape=(4, window_size, window_size), init='glorot_uniform'))
    model.add(Activation('relu'))
      
    # #layer 2 :
    model.add(MaxPooling2D(pool_size=(3, 3) , strides = (3, 3)))
      
      
    #layer 3 :
    model.add(Convolution2D(64, 3, 3, border_mode='valid', init='glorot_uniform'))
    model.add(Activation('relu'))
      
      
      
    #layer 4 :
    model.add(Flatten())
    model.add(Dense(512, init='glorot_uniform'))
    model.add(Activation('relu'))
       
    #layer 5 :
    model.add(Dense(nb_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))
    
    return model



def compile_model():
    model = big_model()
       
#     optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    sgd = keras.optimizers.SGD(lr=1e-4, decay=1e-10, momentum=0.9, nesterov=True)
    print ("--compiling model")
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    print ("--model compiled")
    
    return model
    
def setup_model():
    
    print ("======> warning : you called setup_model from model_setup, previous model is gone <======")
    
    choise = raw_input("do you really want to do that ? type \"yes I do !\" : ")
    if(choise != "yes I do !"):
        return
    
    model = compile_model()



    json_string = model.to_json()
    open(file_model, 'w').write(json_string)
    model.save_weights(file_param, overwrite=True)
    
    print ("============> model setup done <============")





if __name__ == "__main__":
    setup_model()
    
   
    

    
    
  

































