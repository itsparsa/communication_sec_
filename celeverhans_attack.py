import subprocess
subprocess.call(['pip', 'install', "cleverhans"])
import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, Conv2D

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

        train = [t[:128*1000].reshape(-1,batch,*t.shape[1:]) for t in [a.X_train,a.y_train]]
        test = [t[:128*200].reshape(-1,batch,*t.shape[1:]) for t in [a.X_test,a.y_test]]

        data = EasyDict(train=train, test=test)
        # Load training and test data
        model = a.model
        loss_object =tf.keras.losses.MeanSquaredError()
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        # Metrics to track the different accuracies.
        train_loss = tf.metrics.Mean(name="train_loss")
        test_acc = tf.keras.metrics.CategoricalAccuracy()
        test_acc_fgsm = tf.keras.metrics.CategoricalAccuracy()
        test_acc_pgd = tf.keras.metrics.CategoricalAccuracy()

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = model(x)
                loss = loss_object(y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)

        print("Train model with adversarial training")
        # Train model with adversarial training
        for epoch in range(FLAGS["nb_epochs"]):
            # keras like display of progress
            progress_bar_train = tf.keras.utils.Progbar(50000)
            for i in range(data.train[0].shape[0]):
                if FLAGS['adv_train']:
                    # Replace clean example with adversarial example for adversarial training
                    x = projected_gradient_descent(model, data.train[0][i,:], FLAGS['eps'], 0.01, 40, np.inf)
                train_step(data.train[0][i,:], data.train[1][i,:])
                progress_bar_train.add(data.train[0][i,:].shape[0], values=[("loss", train_loss.result())])

        print("Evaluate on clean and adversarial data")
        print("getting clean data from model")
        
        # Evaluate on clean and adversarial data
        def get_clean_data(a,limit):
              pred = a.model.predict(a.X_train)
              if a.name == "deepmimo":
                diff = np.mean(np.mean(np.mean(np.abs(pred - a.y_train),axis=-1),axis=-1),axis=-1)
              else : diff = np.mean(np.abs(pred - a.y_train),axis=-1)
              best_answer_index = np.argsort(diff)
              return dict(X = a.X_train[best_answer_index][:limit] , y = a.y_train[best_answer_index][:limit])

        clean = get_clean_data(a)

        progress_bar_test = tf.keras.utils.Progbar(10000)
        for i in range(clean["X"].shape[0]):
            y_pred = model(clean["X"][i,:])
            test_acc(clean["y"][i,:], y_pred)

            x_fgm = fast_gradient_method(model, clean["X"][i,:], FLAGS.eps, np.inf)
            y_pred_fgm = model(x_fgm)
            test_acc_fgsm(clean["y"][i,:], y_pred_fgm)

            x_pgd = projected_gradient_descent(model, clean["X"][i,:], FLAGS.eps, 0.01, 40, np.inf)
            y_pred_pgd = model(x_pgd)
            test_acc_pgd(clean["y"][i,:], y_pred_pgd)

            progress_bar_test.add(clean["X"][i,:].shape[0])
        
        print(a.name)    

        print(
            "test acc on clean examples (%): {:.3f}".format(test_acc.result() * 100)
        )
        print(
            "test acc on FGM adversarial examples (%): {:.3f}".format(
                test_acc_fgsm.result() * 100
            )
        )
        print(
            "test acc on PGD adversarial examples (%): {:.3f}".format(
                test_acc_pgd.result() * 100
            )
        )

        return dict(acc = test_acc.result(),acc_FGM = test_acc_fgsm.result(),acc_pdg = test_acc_pgd.result())

