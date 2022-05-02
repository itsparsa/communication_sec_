import tensorflow as tf
import numpy as np
from keras.layers.core import Dense
from keras.layers import Conv1D,Flatten, BatchNormalization, Input
from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from generate_model import Designed_model as GM
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Reshape,Dense, GaussianNoise,Lambda,Dropout
from keras.models import Model
from keras import regularizers
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam,SGD
from keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
class MIST(GM):
   def __init__(self,data_path='',k = 500 ,
                      rate = 1/2 ,
                      SNR_dB_start_Eb = -1 ,
                      SNR_dB_stop_Eb = 7 ,
                      SNR_points = 9 ,
                      n_samples = 10 , 
                      train_batch = 1024,
                      test_batch = 1024,
                      g1=[1,1,1],
                      g2=[1,0,1]):
      super(MIST, self).__init__(data_path)
      #dataword length  => k  
      #encoder rate is used => rate 
      self.N = int(k/rate)  #codeword length
      for i,(key,value) in enumerate(locals().items()):
          if i < 1 : continue
          setattr(self,key,value)
      self.name = "mist"
      self.loss = tf.keras.losses.MeanSquaredError()
      self.loss_get_clean = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
      self.metric = self.ber
      self.loss_fn = self.create_loss_fn()

   def build_model(self):
      self.model = self.MIST_model(input_shape=(self.N,1),output_shape=(self.k)) 
      self.compile_model() 
      return self.model

   def build_model_and_get_data(self):
      self.build_model()
      self.X_train , self.y_train = self.get_data()
      self.compile_model() 
   
   def train_gpu(self,epochs = 200 , batch_size=1024): 
       with tf.device(tf.test.gpu_device_name()):
          self.train(epochs=epochs, batch_size=batch_size)    
 
   def train(self, epochs = 200, batch_size=1024 ):
      self.n_samples = int(epochs/self.SNR_points) 
      self.build_model()
      self.compile_model()
      history = self.MIST_train_idea()
      return history
   
   
   def get_data(self , limit = None ) :
      sigmas = self. get_sigmas(self.SNR_dB_start_Eb,
                                self.SNR_dB_stop_Eb,
                                self.SNR_points,
                                self.k,
                                self.N)
      
      X,y = self.generate_uncode_noisy_signal(sigmas)
      return  [X,y]

  
   

   
   def generate_uncode_noisy_signal(self,sigmas):
     
     X = np.zeros([len(sigmas),self.n_samples,self.train_batch,self.N,1])
     y = np.zeros([len(sigmas),self.n_samples,self.train_batch,self.k])
     for i in range(0,self.n_samples): 
        for ii in range(0,len(sigmas)):
            #Generating dataword and codeword
            uncoded = np.random.randint(0,2,size=(self.train_batch,self.k))
            encoded = np.zeros([uncoded.shape[0], self.N])
            # memory 2 convolutional encoder used g1, g2
            for iii in range(0,self.train_batch): 
                encoded[iii,:] = self.convenc (uncoded[iii,:],self.g1,self.g2,self.k)
            #Modulate
            signal = 2*encoded - 1
            #Adding noise
            noisy_signal = signal + np.random.normal(0, sigmas[i], size= np.shape(signal))
            ## train 
            X[i,ii,:,:,0] = noisy_signal
            y[i,ii,:,:] = uncoded
      
     return [ X.reshape(-1,self.N) , y.reshape(-1,self.k)]




   def MIST_train_idea(self , limit = None ):
     sigmas = self.get_sigmas(self.SNR_dB_start_Eb,
                                self.SNR_dB_stop_Eb,
                                self.SNR_points,
                                self.k,
                                self.N)
     history = []
     self.X_train = np.zeros([len(sigmas),self.n_samples,self.train_batch,self.N,1])
     self.y_train = np.zeros([len(sigmas),self.n_samples,self.train_batch,self.k])
     for i in range(0,len(sigmas)):
       print("SNR  {}/{}".format(i,len(sigmas)))
       for ii in range(0,self.n_samples):
            #Generating dataword and codeword
            uncoded = np.random.randint(0,2,size=(self.train_batch,self.k))
            encoded = np.zeros([uncoded.shape[0], self.N])
            # memory 2 convolutional encoder used g1, g2
            for iii in range(0,self.train_batch): 
                encoded[iii,:] = self.convenc(uncoded[iii,:],self.g1,self.g2,self.k)
            #Modulate
            signal = 2*encoded - 1
            #Adding noise
            noisy_signal = signal + np.random.normal(0, sigmas[i], size= np.shape(signal))
            ## train l
            print("epoch  {}/{}".format(ii,self.n_samples))
            self.X_train[i,ii,:,:,0] = noisy_signal
            self.y_train[i,ii,:,:] = uncoded
            x_train = np.expand_dims(noisy_signal,axis=2)
            history.append( self.model.fit(x_train,uncoded, epochs = 1, batch_size = self.train_batch , verbose=2))

     return history 

   def compile_model(self, optimizer = 'adam',loss = 'mse' ):            
      self.model.compile(optimizer=optimizer, loss=loss, metrics=[self.ber]) 
      return self.model

   
   @staticmethod
   def get_sigmas(SNR_dB_start_Eb,SNR_dB_stop_Eb,SNR_points,k,N):
      
      SNR_dB_start_Es = SNR_dB_start_Eb + (10*np.log10(1.0*k/N))
      SNR_dB_stop_Es = SNR_dB_stop_Eb + (10*np.log10(1.0*k/N))
      SNR_range=np.linspace(SNR_dB_start_Es, SNR_dB_stop_Es, SNR_points)
      return np.sqrt(1/(2*10**(SNR_range/10)))
   
   
   @staticmethod
   def convenc(data,g1,g2,k):
    enc_msg = np.zeros([2*k])
    enc_msg[0::2] = (np.convolve(data,g1)%2)[0:k]
    enc_msg[1::2] = (np.convolve(data,g2)%2)[0:k]
    return enc_msg   

    
   @staticmethod
   def ber(y_true, y_pred):
     return  K.mean(K.cast(K.not_equal(y_true, K.round(y_pred)),dtype='float32'))
   
   @staticmethod
   def MIST_model(input_shape , output_shape):  
      input_batch=Input(shape=input_shape)
      conv1 = Conv1D(10, 24, activation='relu',padding='same')(input_batch)
      batch_norm1=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
      conv2 = Conv1D(50, 24, activation='relu',padding='same')(batch_norm1)
      batch_norm2=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
      conv3 = Conv1D(50, 24, activation='relu',padding='same')(batch_norm2)
      batch_norm3=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
      flatten = Flatten()(batch_norm3)
      msg_out = Dense(output_shape, activation='sigmoid')(flatten)

      return Model(inputs=input_batch,outputs=msg_out)
  