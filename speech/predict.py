import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import load_model
#from scipy.io import wavfile 
#from math import log

tf.reset_default_graph()
K.clear_session()

# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.7
 
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

filename = "my_0.npy"



autoencoder = load_model("aecd_snd")

eval_in = np.load("inae/"+filename)

tmp = autoencoder.predict(eval_in)
np.save("out/"+filename,tmp)
del autoencoder