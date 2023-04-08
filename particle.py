import numpy as np
from copy import deepcopy
import CHOCNN.utils
import keras.backend
import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Add, Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Activation, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.advanced_activations import ReLU
from keras import regularizers 
from keras.optimizers import Adam, Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

class Particle:
    def __init__(self,p0, input_width, input_height, input_channels, max_conv_kernel, max_out_ch, max_pool_kernel, max_fc_neurons, output_dim):
                      
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.feature_width = input_width
        self.feature_height = input_height

        self.max_conv_kernel = max_conv_kernel
        self.max_out_ch = max_out_ch
        self.max_pool_kernel = max_pool_kernel
        
        self.max_fc_neurons = max_fc_neurons
        self.output_dim = output_dim

        self.layers = []
        self.acc = None
        self.acc_train = None
        self.vel = [] # Initial velocity
        self.pBest = []
        self.p0 = p0

        self.parameters = None

        # Build particle architecture
        self.initialization()
        
        # Update initial velocity
        self.vel = self.layers
        
        self.model = None
        self.pBest = deepcopy(self)

    
    def __str__(self):
        string = ""
        for z in range(len(self.layers)):
            string = string + self.layers[z]["type"] + " | "
        
        return string

    def initialization(self):
      if self.p0 == 0 :
        # particle0 ,block1
        self.layers = utils.add_convinit(self.layers, 64,3)
        self.layers = utils.add_convinit(self.layers, 64,3)
        self.layers = utils.add_poolinit(self.layers)

        self.layers = utils.add_convinit(self.layers, 128,3)
        self.layers = utils.add_convinit(self.layers, 128,3)
        self.layers = utils.add_poolinit(self.layers)
        
        self.layers = utils.add_convinit(self.layers, 256,3)
        self.layers = utils.add_convinit(self.layers, 256,3)
        self.layers = utils.add_poolinit(self.layers)

         
        self.layers = utils.add_fc(self.layers, self.max_fc_neurons)
        self.layers = utils.add_fc(self.layers, self.output_dim) 

      else:

        for i in range(1, 4):
          self.layers = utils.add_conv(self.layers, self.max_out_ch, self.max_conv_kernel)
          self.layers = utils.add_conv(self.layers, self.max_out_ch, self.max_conv_kernel)
          self.layers = utils.add_pool(self.layers, self.max_pool_kernel)
         
        self.layers = utils.add_fc(self.layers, self.max_fc_neurons) 
        self.layers = utils.add_fc(self.layers, self.output_dim)   
            
    

    def velocity(self, gBest):
        self.vel = utils.computeVelocity(gBest, self.pBest.layers, self.layers)

    def update(self):
        new_p = utils.updateParticle(self.layers, self.vel)
        # new_p = self.validate(new_p)
        self.layers = new_p
        self.model = None

   
    ##### Model methods ####
    def model_compile(self, dropout_rate):
        list_layers = self.layers
        self.model = Sequential()

        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "conv":
                n_out_filters = list_layers[i]["ou_c"]
                kernel_size = list_layers[i]["kernel"]

                if i == 0:
                  in_w = self.input_width
                  in_h = self.input_height
                  in_c = self.input_channels
                  self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", data_format="channels_last", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None, input_shape=(in_w, in_h, in_c)))
                  self.model.add(Activation("relu"))
                
                else:
                  self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                  self.model.add(Activation("relu"))
                  
            if list_layers[i]["type"] == "max_pool":
                # kernel_size = list_layers[i]["kernel"]

                self.model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
                self.model.add(Dropout(dropout_rate))

            elif list_layers[i]["type"] == "avg_pool":
                # kernel_size = list_layers[i]["kernel"]

                self.model.add(AveragePooling2D(pool_size=(3, 3), strides=2, padding='same'))
                self.model.add(Dropout(dropout_rate))

            if list_layers[i]["type"] == "fc":
                if list_layers[i-1]["type"] != "fc":
                    self.model.add(Flatten())
                    

                if i == len(list_layers) - 1:
                    self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', activation=None))
                    self.model.add(Activation("softmax"))
                else:
                    self.model.add(Dense(list_layers[i]["ou_c"], kernel_initializer='he_normal', bias_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), activation=None))                   
                    self.model.add(ReLU())
                    self.model.add(Dropout(dropout_rate))

        # adam = Adam(learning_rate=0.001)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
        self.parameters = int(np.sum([keras.backend.count_params(p) for p in set(self.model.trainable_weights)]))
        self.model.summary()
        return self.parameters

    def model_fit(self, x_train, y_train, batch_size, epochs):
        # TODO: add option to only use a sample size of the dataset
        hist = self.model.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs)
        return hist

    def model_fit_complete(self, x_train, y_train, batch_size, epochs):
        hist = self.model.fit(x=x_train, y=y_train, validation_split=0.0, batch_size=batch_size, epochs=epochs)
        return hist
    
    def model_delete(self):
        # This is used to free up memory during PSO training
        del self.model
        keras.backend.clear_session()
        # tf.reset_default_graph()
        self.model = None
