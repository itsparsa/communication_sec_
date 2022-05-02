import keras
from keras.models import Model
import numpy as np
import os
import tensorflow as tf
import keras.backend as k


class Designed_model():
   def __init__(self,data_path) :
      self.data_path = data_path
      self.X_train = None
      self.y_train = None
      self.X_test = None
      self.y_test = None
      self.model = None 
      self.name = None
      self.loss = None
      self.loss_get_clean = None
      self.metric = None 
      self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)
      self.loss_fn = self.create_loss_fn()

   def build_model_and_get_data(self):
     pass
   
   def get_data(self):
      pass
   
   def train(self):
      pass

   def delete_data_buffers(self):
      del self.X_train 
      del self.y_train
      del self.X_test
      del self.y_test 

   
   def train_gpu(self,epochs = 200 , batch_size=1024 , get_data_build =True): 
       with tf.device(tf.test.gpu_device_name()):
          self.train(epochs, batch_size ,get_data_build)

   def train(self, epochs =20,batch_size=1024 , get_data_build =True):
      if get_data_build : self.build_model_and_get_data()
      history = self.model.fit(self.X_train,self.y_train,
                            validation_data=(self.X_test, self.y_test),
                            epochs=epochs, batch_size = batch_size, verbose=2 )
      
      return history   


   def get_clean_data(self,limit):
              pred = self.model(self.X_train)
              diff = self.loss_get_clean(self.y_train,pred).numpy()
              best_answer_index = np.argsort(diff)
              return [self.X_train[best_answer_index][:limit] , self.y_train[best_answer_index][:limit]] 
   
   def create_loss_fn(self):
     temp = self.loss_get_clean
     def loss_fn(labels,logits):
          return temp(labels,logits)
     return loss_fn

   def compile_model(self):
      self.model.compile(
                          optimizer=self.opt,
                          loss= self.loss,
                          metrics=self.metric,
                      )  
      return self.model



   def save(self,data_path):
     path = data_path + '/' +self.name
     i = 0 
     while os.path.isdir(path):
       path = data_path + '/' + self.name + str(i)
       i+=1
     os.mkdir(path) 
     self.model.save_weights(path + '/' +'myModel.h5')
     np.savez_compressed( path + '/train_shape', X_train_shape= np.array(self.X_train.shape) , y_train_shape = np.array(self.y_train.shape))
     print("train data shape saved")
     np.savez_compressed( path + '/train', X_train= self.X_train.reshape(-1,1), y_train = self.y_train.reshape(-1,1))
     print("train data saved")
     if self.name is not 'mist':
      np.savez_compressed( path + '/test_shape', X_test_shape= np.array(self.X_test.shape) , y_test_shape = np.array(self.y_test.shape))
      print("test data shape saved")
      np.savez_compressed( path + '/test', X_test= self.X_test.reshape(-1,1), y_test = self.y_test.reshape(-1,1))
      print("test data saved")

   @classmethod
   def load_from_dir(cls,path,data_path):
      new_model = cls(data_path)
      new_model.build_model_and_get_data()
      new_model.model = new_model.models.load_weights(path+"/myModel.h5")
      loaded = np.load(path + '/train.npz')
      loaded_shape = np.load( path + '/train_shape.npz')
      X_train_shape , y_train_shape = loaded_shape["X_train_shape"] , loaded_shape["y_train_shape"] 
      new_model.X_train , new_model.y_train = loaded["X_train"].reshape(X_train_shape) , loaded["y_train"].reshape(y_train_shape)
      del loaded
      if a.name is not 'mist':
        loaded = np.load(path + '/test.npz')
        loaded_shape = np.load( path + '/test_shape.npz')
        X_test_shape , y_test_shape = loaded_shape["X_test_shape"] , loaded_shape["y_test_shape"] 
        new_model.X_test , new_model.y_test = loaded["X_test"].reshape(X_test_shape) , loaded["y_test"].reshape(y_test_shape)
   
   @staticmethod
   def ber(y_true, y_pred):
      return  k.mean(k.cast(k.not_equal(y_true, k.round(y_pred)),dtype='float32'))

   def eval(self):
     self.model.evaluate(self.X_test,self.y_test)

   @staticmethod 
   def load(path):
      new_model = self.build_model_and_get_data()
      new_model.model = keras.models.load_model(path+"_model.h5")
      loaded = np.load(path + 'train.npz')
      loaded_shape = np.load( path + 'train_shape.npz')
      X_train_shape , y_train_shape = loaded_shape["X_train_shape"] , loaded_shape["y_train_shape"] 
      new_model.X_train , new_model.y_train = loaded["X_train"].reshape(X_train_shape) , loaded["y_train"].reshape(y_train_shape)
     
   @staticmethod 
   def limit_check(lim,X):
       if lim is not None : 
         if lim < X.shape[0] :
            return True 
         else:
            print("maximum number of input is {}".format(X.shape[0]))  
       return False 
    
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