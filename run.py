import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
import argparse
import numpy as np

(_,_),(x_test,_) = mnist.load_data()
model = tf.keras.models.load_model('src/mnist_reconstruction_AE.h5')
x_test  = x_test/255 

def make_prediction(idx):
    img = np.expand_dims(x_test[idx], axis=-1)[np.newaxis, ...]
    pred = model.predict(img,verbose=0)
    return np.squeeze(pred)

def plot_img(idx):
    _, axes = plt.subplots(1, 2, figsize=(3, 3))
    prediction = make_prediction(idx)
    axes[0].imshow(prediction, cmap='gray')
    axes[1].imshow(x_test[idx], cmap='gray')
    axes[0].set_title('generated')
    axes[1].set_title('original')
    axes[0].axis('off')
    axes[1].axis('off')

parser = argparse.ArgumentParser("plotting result")
parser.add_argument('--idx', type=int)

args = parser.parse_args()

plot_img(args.idx)
plt.show()