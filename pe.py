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
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras import layers , Input
from tensorflow.keras.layers import Dense , LSTM ,Dropout , Activation ,Reshape


class PE(GM):
   def __init__(self,data_path,train_snr):
      super(PE, self).__init__(data_path)
      self.name = 'pe'
      self.train_snr = train_snr


   def build_model_and_get_data(self):
      if type(self.X_train) == type(None) : self.set_data() 
      self.model = self.RNN_model ( self.X_train[0].shape, 
                                    self.y_train[0].shape[0])
      
      self.compile_model(self.model)
      return 
   
  
   
   def norm(self):
      self._mean = self.mean_data(np.array([self.mean_data(self.X_train),
                                          self.mean_data(self.X_test)]))
      self._scale = max(self.scale(self.X_train),
                        self.scale(self.X_test))
      for value in [self.X_train,self.X_test]:
        value = self.transform(value,self._mean,self._scale)

   
   def set_data(self , limit = 100000 , execute = False )  -> None:

      train_batch_size = 128
      train_snr = self.train_snr
      test_snr = np.arange(0, 6.5, 0.5)
      train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
      epoch_setting = np.array([10**1, 10**2, 10**3, 10**4, 10**5])

      if  execute : 
        os.mkdir(self.data_path)
        exec(open(self.data_path + 'deep-neural-network-decoder/RNN/noise/K_16_N_32/"train_data_10^6"/get_data.py'))

        ##### execute train ##########
        X_train, y_train = self.get_data_by_ratio (train_ratio = 0.8 ,
                                                   snr = train_snr,status = "train",
                                                   limit= limit,
                                                   data_path = self.data_path + "data/")    
        
        np.savez_compressed( self.data_path + 'train_shape', X_train_shape= np.array(X_train.shape) , y_train_shape = np.array(y_train.shape))
        print("train data shape saved")
        np.savez_compressed( self.data_path + 'train', X_train= X_train.reshape(-1,1), y_train = y_train.reshape(-1,1))
        print("train data saved")
        del X_train, y_train

        ##### execute test ##########
        X_test, y_test = self.get_data_by_ratio (train_ratio = 0.8 ,
                                                   snr = train_snr,status = "train",
                                                   limit= int(limit*0.2),
                                                   data_path = self.data_path + "data/",
                                                 ) 

        np.savez_compressed( self.data_path + 'test_shape', X_test_shape= np.array(X_test.shape) , y_test_shape = np.array(y_test.shape))
        print("test data shape saved")
        np.savez_compressed( self.data_path + 'test', X_test= X_test.reshape(-1,1), y_test = y_test.reshape(-1,1))
        print("test data saved")
        del X_test, y_test

      ###### Load data from npy saved based on exceuting data earlier ########
      print("Loading Train Data...")
      loaded = np.load( self.data_path + 'train.npz')
      loaded_shape = np.load( self.data_path + 'train_shape.npz')
      X_train_shape , y_train_shape = loaded_shape["X_train_shape"] , loaded_shape["y_train_shape"] 
      self.X_train , self.y_train = loaded["X_train"].reshape(X_train_shape) , loaded["y_train"].reshape(y_train_shape)


      print("Loading test Data...")
      loaded = np.load( self.data_path + 'test.npz')
      loaded_shape = np.load( self.data_path + 'test_shape.npz')
      X_test_shape , y_test_shape = loaded_shape["X_test_shape"] , loaded_shape["y_test_shape"] 
      self.X_test , self.y_test = loaded["X_test"].reshape(X_test_shape) , loaded["y_test"].reshape(y_test_shape)
      print("loading is finished ")
      del loaded
      # make the data independent of snr 
      self.X_train , self.y_train = self.X_train.reshape(-1,X_train_shape[-1]) , self.y_train.reshape(-1,y_train_shape[-1])
      self.X_test , self.y_test = self.X_test.reshape(-1,X_test_shape[-1]) , self.y_test.reshape(-1,y_test_shape[-1])
      self.norm ()
   
   
   @staticmethod
   def transform(x,mean,scale):
     return (x-mean)/scale

   @staticmethod
   def mean_data(data):
      mean = np.mean(data,axis=0)
      for i in range(1,len(data.shape)-2):
        mean = np.mean(mean,axis=0)   
      return mean

   @staticmethod 
   def scale(data):
      max = np.max(np.abs(data))
      return max 



   @staticmethod
   def compile_model(model):
      model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss= tf.keras.losses.MeanSquaredError(),
                    metrics=["accuracy"],
                )
      return model
   

   @staticmethod
   def RNN_model (input_shape,output_shape, Dropout_rate = 0.1) :
  
    input_signal = Input(shape = input_shape)
    reshape = Reshape((*input_shape,1))(input_signal)
    lstm = LSTM(256,
                dropout = Dropout_rate,
                kernel_initializer= tf.keras.initializers.GlorotNormal(seed=None))(reshape)

    output = Dense(output_shape,"sigmoid")(lstm)
    
    return Model(inputs = input_signal, outputs = output)


   @staticmethod
   def get_data_by_ratio (train_ratio, snr, status = "train", limit= None,
                      data_path = "/content/project_3/PE/data/"):
      
      if status == "train" : data = sio.loadmat(data_path+"ratio_0.4_train_snr_8dB")
      elif status == "test" : data = sio.loadmat(data_path+'test_snr_0.0dB')
      x_shape = data['x_'+status].shape
      y_shape = data['y_'+status].shape
      x_shape = [min(x_shape[0],limit) , x_shape[1]]
      y_shape = [min(y_shape[0],limit) , y_shape[1]]

      x = np.zeros([len(snr),*x_shape])
      y = np.zeros([len(snr),*y_shape])
      print(x_shape)
      del data
    
      for i, tr_snr in enumerate(snr):
          print("added {} SNR".format(tr_snr))
          if status == "train" : filename = 'ratio_' + str(train_ratio) + '_train_snr_' + str(tr_snr) + 'dB'
          elif status == "test" : filename = 'test_snr_' + str(tr_snr) + 'dB'
          data = sio.loadmat(data_path+filename)
          x[i,:,:] = data['x_'+status][:limit,:]
          y[i,:,:] = data['y_'+status][:limit,:]

      return [x , y]




