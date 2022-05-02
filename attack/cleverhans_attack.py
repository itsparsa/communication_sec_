import subprocess
subprocess.call(['pip', 'install', "cleverhans"])
import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, Conv2D
from copy import deepcopy
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method


class Attack_handler:
    def __init__(self,nb_epochs= 1,eps = 0.05, adv_train = False ,batch = 128 ):
       for i,(key,value) in enumerate(locals().items()):
          if i < 1 : continue
          setattr(self,key,value)

    
    def attack(self,a,clean_limit=100):
      FLAGS = {"nb_epochs": self.nb_epochs ,"eps": self.eps,"adv_train": self.adv_train}
      self.start_attack(a,FLAGS,self.batch,clean_limit)


    
    @staticmethod
    def start_attack(a , FLAGS = {"nb_epochs": 1,"eps": 0.05,"adv_train": False} ,batch = 128,limit =100):

        lim = int(a.X_train.shape[0]/128)
        train = [t[:lim*batch].reshape(lim,batch,*t.shape[1:]) for t in [a.X_train,a.y_train]]
        #train = [a.X_train[:100],a.y_train[:100]]
        # Load training and test data
        model_copy= keras.models.clone_model(a.model1)
        model_copy.build(a.X_train[0]) # replace 10 with number of variables in input layer
        model_copy.compile(optimizer= a.opt, loss=a.loss)
        model_copy.set_weights(a.model.get_weights())
        
        model = model_copy
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Metrics to track the different accuracies.
        train_loss = tf.metrics.Mean(name="train_loss")
        test_acc = deepcopy(a.metric)
        test_acc_fgsm = deepcopy(a.metric)
        test_acc_pgd = deepcopy(a.metric)
        
        loss_fn = a.loss_fn


        clean = a.get_clean_data(limit)
        lim = int(clean[0].shape[0]/128)
        test = [t[:lim*batch].reshape(lim,batch,*t.shape[1:]) for t in clean]
        #test = clean
        data = EasyDict(train=train, test=test)

        print(" data train [0]",data.train[0].shape)
        print(" data train [1]",data.train[1].shape)
        print(" data test [0]",data.test[0].shape)
        print(" data test [1]",data.test[1].shape)

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = loss_object(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

        print("Train model with adversarial training")
        
        #print(model(data.train[0][0]))
        # Train model with adversarial training
        for epoch in range(FLAGS["nb_epochs"]):
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(50000)
            for i in range(data.train[0].shape[0]):
                x = data.train[0][i]
                y = data.train[1][i]
                if FLAGS['adv_train']:
                    # Replace clean example with adversarial example for adversarial training
                    x = projected_gradient_descent(model, x, FLAGS['eps'], 0.01, 40, np.inf,loss_fn=loss_fn,y=y)
                train_step(x,y)
                progress_bar_train.add(x.shape[0], values=[("loss", train_loss.result())])

        print("Evaluate on clean and adversarial data")
        print("getting clean data from model")
        
        # Evaluate on clean and adversarial data
        # print(data.test[1][0,0:2])
        # print(model(data.test[0][0,0:2]))

        
        progress_bar_test = tf.keras.utils.Progbar(50000)
        for i in range(data.test[0].shape[0]):
            x = data.test[0][i,:]
            y = data.test[1][i,:]
            y_pred = model(x)
            print(y.shape,y_pred.shape)
            test_acc(y, y_pred)
            x_fgm = fast_gradient_method(model, x, FLAGS['eps'], np.inf,loss_fn=loss_fn, y=y,)
            y_pred_fgm = model(x_fgm)
            test_acc_fgsm(y, y_pred_fgm)

            x_pgd = projected_gradient_descent(model, x,FLAGS['eps'], 0.01, 40, np.inf,loss_fn=loss_fn, y=y )
            y_pred_pgd = model(x_pgd)
            test_acc_pgd(y, y_pred_pgd)

            progress_bar_test.add(x.shape[0])
        
        print(a.name)    

        print(
            "test acc on clean examples {} : {:.3f}".format(a.loss.name,test_acc.result())
        )
        print(
            "test acc on FGM adversarial examples {}: {:.3f}".format(a.loss.name,
                test_acc_fgsm.result()
            )
        )
        print(
            "test acc on PGD adversarial examples {} : {:.3f}".format(a.loss.name,
                test_acc_pgd.result()
            )
        )
        info = vars(a)
        return dict(acc = test_acc.result(),acc_FGM = test_acc_fgsm.result(),acc_pdg = test_acc_pgd.result(),Network=info)








