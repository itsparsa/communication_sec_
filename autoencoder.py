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

class aphl(GM):
   def __init__(self,data_path,num_train_limit=None):
      super().__init__(data_path)
      self.name ='aphl'
      self.num_train_limit = num_train_limit
      self.metric = tf.keras.metrics.CategoricalCrossentropy()
      self.loss = tf.keras.losses.CategoricalCrossentropy()
      self.loss_get_clean = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
      self.loss_fn = self.create_loss_fn()
   
   def build_model_and_get_data(self):
      if type(self.X_train) == type(None) : self.set_data(limit=self.num_train_limit) 
      self.model = self.CNN_model_article(
         self.X_train[0].shape, 
         self.y_train[0].shape[0])
      self.compile_model()
      return 
  
   
   def set_data(self , limit = None) -> None:
      ##read from directory
      df = pd.read_pickle( self.data_path + "RML2016.10a_dict.pkl")
      d = pd.DataFrame(df.keys())
      snrs,mods = d[0].unique() , d[1].unique().tolist()
      if type (snrs[1]) == str : 
         snrs,mods = d[1].unique() , d[0].unique().tolist()
      ## creat label 
      X = []  
      label = []
      for key , value in df.items():
         label.extend([key]*value.shape[0])
         X.append(value)
      X = np.vstack(X)
      
      ## make one-hot label
      encoded = to_categorical([i for i in range(len(mods))]) 
      
      # seperate data for train and test
      y = label 
      
      if self.limit_check(limit,X):
            X = X[:limit]
            y = y[:limit] 

      X_mean = self.mean_data(X)
      self._mean = X_mean
      
      X_scale = self.scale(X)
      self._mean = X_scale
      
      X = self.transform(X,X_mean,X_scale)

      X_train, X_test, y_train, y_test = train_test_split(
                                             X, y, test_size=0.4, random_state=42)

      SNR_train = np.array([lbl[1] for lbl in y_train])
      y_train = np.array([self.mods_to_one_hot(mods,lbl[0],encoded) for lbl in y_train])
      y_test  = np.array([self.mods_to_one_hot(mods,lbl[0],encoded) for lbl in y_test])

      sorted_arg_SNR_descended = SNR_train.argsort()[::-1]
      X_train = X_train[sorted_arg_SNR_descended]
      y_train = y_train[sorted_arg_SNR_descended]

      X = X.astype('float64')
      y_train = y_train.astype('float64')
      y_test = y_test.astype('float64')

      self.X_train , self.y_train, self.X_test, self.y_test = X_train ,y_train, X_test,y_test
   
   
   @staticmethod
   def mods_to_one_hot (mods,inp,encoded):
      return encoded[mods.index(inp)]


   @staticmethod
   def complex_to_array(dataset):
      shape = dataset.shape
      temp = dataset.reshape([*shape,1])
      real_num , imag_num = np.real(temp) , np.imag(temp)
      array_form_of_complex = np.concatenate((real_num,imag_num),axis=len(shape))
      return array_form_of_complex 



   
   @staticmethod
   def CNN_model_article(input_shape = (2,128) , output_shape=11):

    input_signal = Input(shape = input_shape)
    reshape = Reshape((2,128,1))(input_signal)
    conv_1 = Conv2D(128 , kernel_size = (2, 8), activation='relu' )(reshape)
    max_pool_1 = MaxPooling2D (pool_size=(1,2), strides = 2,data_format ='channels_last')(conv_1)
    
    conv_2 = Conv2D(64, kernel_size =(1, 16), activation='relu')(max_pool_1)
    max_pool_2 = MaxPooling2D (pool_size=(1,2), strides = 2)(conv_2)

    flatten = Flatten()(max_pool_2)

    dense_1 = Dense(128, activation='relu')(flatten)
    dense_2 = Dense(64, activation='relu')(dense_1)
    dense_3 = Dense(32, activation='relu')(dense_2)
    dense_4 = Dense(output_shape, activation='softmax')(dense_3)

    return Model(inputs = input_signal, outputs = dense_4)  