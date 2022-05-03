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
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class PE(GM):
   def __init__(self,data_path, 
                     limit = 10000,
                     execute = {"status":False,"limit":10000,"train_ratio":0.8 ,
                     "train_snr" :np.arange(20, -4, -2)},
                      ):
      super(PE, self).__init__(data_path)
      self.name = 'pe'
      self.limit = limit 
      self.execute = execute 
      self.loss = tf.keras.losses.MeanSquaredError()
      self.loss_get_clean = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
      self.metric = self.ber_function()
      self.loss_fn = self.create_loss_fn()

   def build_model_and_get_data(self):
      if type(self.X_train) == type(None) : self.set_data(self.limit, self.execute) 
      self.model = self.RNN_model ( self.X_train[0].shape, 
                                    self.y_train[0].shape[0])
      
      self.compile_model()
      return 
   
   def set_data(self , limit = 100000 ,  execute ={"status":False,
                                                  "limit":10000,
                                                  "train_ratio":0.8 ,
                                                  "train_snr" :np.arange(20, -4, -2)})  -> None:


      #TODO 
      #test_snr = np.arange(0, 6.5, 0.5)
      #train_batch_size = 128

      if  execute["status"] : 
        os.mkdir(self.data_path)
        exec(open(self.data_path + 'deep-neural-network-decoder/RNN/noise/K_16_N_32/"train_data_10^6"/get_data.py'))

        ##### execute train ##########
        X_train, y_train = self.get_data_by_ratio (train_ratio = execute["train_ratio"] ,
                                                   snr =  execute["train_snr"],
                                                   status = "train",
                                                   limit= execute['limit'],
                                                   data_path = self.data_path + "data/")    
        
        np.savez_compressed( self.data_path + 'train_shape', X_train_shape= np.array(X_train.shape) , y_train_shape = np.array(y_train.shape))
        print("train data shape saved")
        np.savez_compressed( self.data_path + 'train', X_train= X_train.reshape(-1,1), y_train = y_train.reshape(-1,1))
        print("train data saved")
        del X_train, y_train

        ##### execute test ##########
        X_test, y_test = self.get_data_by_ratio (train_ratio = execute["train_ratio"],
                                                   snr =  execute["train_snr"],
                                                   status = execute["limit"],
                                                   limit= int( execute['limit']*( 1 - execute["train_ratio"])),
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
      self.X_train , self.y_train = self.X_train[:limit] ,self.y_train[:limit]
      self.X_test , self.y_test = self.X_test[:limit] ,self.y_test[:limit]
      self.norm ()
   
   def norm(self):
      self._mean = self.mean_data(np.array([self.mean_data(self.X_train),
                                          self.mean_data(self.X_test)]))
      self._scale = max(self.scale(self.X_train),
                        self.scale(self.X_test))
      for value in [self.X_train,self.X_test]:
        value = self.transform(value,self._mean,self._scale)

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
   def plot_imshow(*atr):
      fig = make_subplots(rows=len(atr), cols=1)
      for j in range(len(atr)):
        for i in range(1):
          fig.add_trace(px.imshow(atr[j]).data[0], row=j+1,col=1)
      fig.show()
      return fig 





