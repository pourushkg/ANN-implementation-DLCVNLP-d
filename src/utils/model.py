import tensorflow as tf
import time
import os
import pandas as pd 
import matplotlib.pyplot as plt

def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES):

    LAYERS = [tf.keras.layers.Flatten(input_shape=[28,28], name="input_layer"),
              tf.keras.layers.Dense(300,activation="relu",name="first_hidden_layer"),
              tf.keras.layers.Dense(100,activation="relu",name="second_hidden_layer"),
              tf.keras.layers.Dense(NUM_CLASSES,activation="softmax",name="output_layer")]
    
    model_clf = tf.keras.models.Sequential(LAYERS)

    model_clf.summary()

    model_clf.compile(loss=LOSS_FUNCTION,
                      optimizer=OPTIMIZER,
                      metrics=METRICS)
    return model_clf

def create_plots(history):
    pd.DataFrame(history.history).plot(figsize=(10,7))
    plt.grid(True)
    plt.show()




def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d-%H%M%S{filename}")

    return unique_filename


def save_model(model,model_name,model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir,unique_filename)

    model.save(path_to_model)



    
