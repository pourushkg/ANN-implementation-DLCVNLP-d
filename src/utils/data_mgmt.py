## this is use for data related things 
import tensorflow as tf 

import matplotlib.pyplot as plt

def get_data(validation_data):
    mnist=tf.keras.datasets.mnist
    (x_train_full,y_train_full),(x_test,y_test)=mnist.load_data()

    x_valid,x_train=x_train_full[:validation_data] / 255,x_train_full[validation_data:] / 255
    y_valid,y_train=y_train_full[:validation_data],y_train_full[validation_data:]

    ## scale the test data as well
    x_test=x_test/255.0

    return (x_train,y_train),(x_valid,y_valid),(x_test,y_test)



