import subprocess
subprocess.call(['pip', 'install', "DeepMIMO"])
import DeepMIMO
import numpy as np 
from generate_model import Designed_model as GM
from sklearn.model_selection import train_test_split
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , Input , Sequential 
from tensorflow.keras.layers import Flatten , Dense , Reshape ,Dropout
import keras.backend as K
from keras.models import Model

class Deep_MIMO(GM):
   def __init__(self,data_path ,active_BS = 64,
                                user_row_first = 1,
                                user_row_last = 10,
                                bandwidth = 0.05,
                                subcarriers = 64 ,
                                subcarriers_sampling = 1 ,
                                subcarriers_limit = 32,
                                num_paths = 1,
                                num_set_antenna_M = 8,
                                MLP_DroupOut = 0.1):
      super(Deep_MIMO, self).__init__(data_path)
      for i,(key,value) in enumerate(locals().items()):
          if i < 1 : continue
          setattr(self,key,value)
      self.name ="deepmimo"
      self.metric = tf.keras.metrics.MeanAbsoluteError()
      self.loss = tf.keras.losses.MeanSquaredError()
      self.loss_get_clean = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
      self.loss_fn = self.create_loss_fn()

   
   def build_model_and_get_data(self):
      if type (self.X_train) == type(None): self.set_data() 
      input_shape = self.X_train[0].shape
      output_shape = self.y_train[0].shape
      self.model = self.MLP_with_dropouts (input_shape , output_shape,dropout_rate= self.MLP_DroupOut)
      self.compile_model ()
      return self.model
   
   ###################### data ##########################
   def set_data(self , limit = None) -> None:

      dataset_up = self.get_MIMO_data('I1_2p4')
      dataset_down = self.get_MIMO_data('I1_2p4')

      dataset_up = self.draw_necessary_inf_in_desired_shape (dataset_up)
      dataset_down = self.draw_necessary_inf_in_desired_shape (dataset_down)

      data_up = self.complex_to_array(dataset_up)
      data_down = self.complex_to_array(dataset_down)

      
      X_mean = self.mean_data(data_up)
      y_mean = self.mean_data(data_down)
      self._mean = X_mean
      
      X_scale = self.scale(data_up)
      y_scale = self.scale (data_down)
      self._scale = X_scale
      
      norm_data_up = self.transform(data_up,X_mean,X_scale)
      norm_data_down = self.transform(data_down,y_mean,y_scale)

      X = np.array([u*self.make_mask(u,num_set_antenna_M = self.num_set_antenna_M) for u in norm_data_up[:]])
      y = norm_data_down

      X = X.astype('float64')
      y = y.astype('float64')
      X_train, X_test, y_train, y_test = train_test_split(
                                              X, y, test_size=0.2, random_state=42)
      
      self.X_train , self.y_train, self.X_test, self.y_test = X_train ,y_train, X_test,y_test

   @staticmethod
   def draw_necessary_inf_in_desired_shape(data):
     #just using one RX antenna data and one TX antenna form all antennas information  
      data = [data[i]['user']["channel"][:][:,0,0] for i in range(len(data))]
      data = np.array(data)
      print(data.shape)
      #changing the data into form of : [num_user,num_antena,num_subchannel]  
      data = np.transpose(data,axes=(1,0,2))
      return data
   
   ######################### model ######################
   @staticmethod
   def MLP(input_shape,output_shape,
                      num_layers = [2**10,2**12,2**12,64*32*2],
                      k_r = None):
     
      num_last_layer = output_shape[0]*output_shape[1]*output_shape[2]
      input_signal = Input(shape = input_shape) 
      flatten =Flatten()(input_signal)
      for i,num in enumerate(num_layers[1:-1]):
        if i==0: dense = Dense(num, activation="relu" , kernel_regularizer = k_r)(flatten)
        dense = Dense(num, activation="relu" , kernel_regularizer = k_r)(dense)

      last_dense = Dense(num_last_layer, activation=None , kernel_regularizer = k_r)(dense)    
      out_put = Reshape(output_shape, input_shape=(last_dense))(last_dense)  
      return Model(inputs=input_signal,outputs=out_put) 
 

   @staticmethod
   def MLP_with_dropouts(input_shape,output_shape,
                      num_layers = [2**10,2**12,2**12,64*32*2],
                      dropout_rate=.1 ,
                      k_r = None):
     
      num_last_layer = output_shape[0]*output_shape[1]*output_shape[2]
      input_signal = Input(shape = input_shape) 
      flatten =Flatten()(input_signal)
      for i,num in enumerate(num_layers[1:-1]):
        if i==0: dense = Dense(num, activation="relu" , kernel_regularizer = k_r)(flatten)
        dense = Dense(num, activation="relu" , kernel_regularizer = k_r)(dense)
        drop = Dropout(dropout_rate)(dense)

      last_dense = Dense(num_last_layer, activation=None , kernel_regularizer = k_r)(drop)    
      out_put = Reshape(output_shape, input_shape=(num_last_layer,))(last_dense)   
      return Model(inputs=input_signal,outputs=out_put) 


   @staticmethod
   def NMSE_loss (y_true , y_pred):
      return tf.cast(K.mean(K.square(y_pred - y_true)/ (2*K.square(y_true))),tf.float64)

   @staticmethod
   def MSE_loss (y_true , y_pred):
      return K.mean(K.square(y_pred-y_true))

   ##################### loss #####################
   def loss_get_clean_data_sp(self,y_true,y_pred):
      y_pred = K.reshape(y_pred,shape = (y_pred.shape[0],-1))
      y_true = K.reshape(y_true,shape = (y_pred.shape[0],-1))
      return self.loss_get_clean(y_true,y_pred)
      
   #overwrite beacasue of the special dimention it has 
   def get_clean_data(self,limit):
      y_pred = self.model(self.X_train)
      y_true = self.y_train
      diff = self.loss_get_clean_data_sp(y_true,y_pred)
      diff = diff.numpy()
      best_answer_index = np.argsort(diff)
      return [self.X_train[best_answer_index][:limit] , self.y_train[best_answer_index][:limit]]

   def create_loss_fn(self):
      temp = self.loss_get_clean_data_sp
      def loss_fn(labels,logits):
            return temp(labels,logits)
      return loss_fn

  ###############################  data ################################
   @staticmethod
   def complex_to_array(dataset):
      shape = dataset.shape
      temp = dataset.reshape([*shape,1])
      real_num , imag_num = np.real(temp) , np.imag(temp)
      array_form_of_complex = np.concatenate((real_num,imag_num),axis=len(shape))
      return array_form_of_complex 

   @staticmethod  
   def make_mask (dataset , inp_mask = None , num_set_antenna_M = 8):
      num_ant ,num_subchannel = dataset.shape[0:2];
      mask = np.zeros(dataset.shape);
      if inp_mask == None : 
          antena_selected_mask = np.random.choice(num_ant,
                                                  num_set_antenna_M,
                                                  replace=False)
      else: antena_selected_mask = np.where( mask == 1)
      mask[ antena_selected_mask] =  1
      return mask


   def get_MIMO_data(self,scenario ) :

     # Load the default parameters
      parameters = DeepMIMO.default_params()
      # Set scenario name
      parameters['scenario'] = scenario
      # Set the main folder containing extracted scenarios
      parameters['dataset_folder'] = r''+ self.data_path
      # To activate the basestations from 1 tiil 64, set
      # each antena has 8 RX 
      parameters['active_BS'] = np.arange(1,self.active_BS + 1)
      # To activate the user rows 1-5 
      # according to practical examination i2_2p4 has 201 in each column 
      parameters['user_row_first'] = self.user_row_first
      parameters['user_row_last'] = self.user_row_last


      # Number of BS antennas in (x, y, z)
      parameters['bs_antenna']['shape'] = np.array([1, 1, 1])
      # To generate channels at 0.02 GHz  bandwidth, set
      parameters['OFDM']['bandwidth'] = self.bandwidth
      # To generate OFDM channels with 64 subcarriers, set
      parameters['OFDM']['subcarriers'] = self.subcarriers
      # To sample first 16 subcarriers by every spacing between each, set
      parameters['OFDM']['subcarriers_sampling'] = self.subcarriers_sampling
      #according to the picture in the article we chioced 32 instead of 16, that was mentioned in TABLE 1
      parameters['OFDM']['subcarriers_limit'] = self.subcarriers_limit
      # To only include 1 strongest paths in the channel computation, set
      parameters['num_paths'] = self.num_paths

      #Note: Since this scenario consists 
      #of only one access point, please set “enable_BS2BSchannels”
      #(MATLAB) or “enable_BS2BS” (Python) to zero in the DeepMIMO 
      #generation parameters. 
      parameters['enable_BS2BS'] = 0

      # Generate data
      dataset = DeepMIMO.generate_data(parameters)

      #delete unnccessary info
        #find keys in dictionary which is unneccessary
      dataset_keys = list(dataset[0].keys())
      dataset_user_keys = list(dataset[0]['user'].keys())
        #delte those keys 
      for i in range(len(dataset)):
          for del_key in [key for key in dataset_keys if key is not 'user']:
              del dataset[i][del_key]
          for del_key in [key for key in dataset_user_keys if key is not 'channel']:
              del dataset[i]['user'][del_key]

      return dataset

   ##########plot ##################################################################   
   @staticmethod
   def plot_imshow(*atr):
    fig = make_subplots(rows=len(atr), cols=2)
    for j in range(len(atr)):
        for i in range(2):
            fig.add_trace(px.imshow(atr[j][:,:,i]).data[0], row=j+1,col=i+1)
    fig.show()
    return fig