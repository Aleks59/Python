import tensorflow as tf
import numpy as np
from scipy.io import wavfile 
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from keras.layers import MaxPooling1D, UpSampling1D, Conv1D, Reshape, Flatten
#from keras.layers import MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.optimizers import Adam

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

file0 = np.load("inae/f3_0.npy")
file1 = np.load("inae/f3_1.npy")
file2 = np.load("inae/f3_2.npy")
file3 = np.load("inae/f3_3.npy")
eval_in = np.load("eval_in.npy")

file_in = []
for data in file0:
    file_in.append(data)
del file0    
for data in file1:
    file_in.append(data)
del file1
for data in file2:
    file_in.append(data)
del file2
for data in file3:
    file_in.append(data)
del file3

file_in = np.array(file_in)

snd_in = Input(shape=(8000,))

dropout_rate = 0.3

#x = Dense(1024, activation='elu')(snd_in)
#x = Dense(512, activation='elu')(x)
#x = Dense(256, activation='elu')(x)
#encoded = Dense(128, activation='tanh')(x)
#
#x = Dense(256, activation='elu')(encoded)
#x = Dense(512, activation='elu')(x)
#x = Dense(1024, activation='elu')(x)
#decoded = Dense(8000, activation = 'tanh')(x)

x = Reshape((8000,1))(snd_in)
x = Conv1D(16, 10, activation='relu', padding = 'same')(x)
x = MaxPooling1D(4)(x)
x = Dropout(dropout_rate)(x)
x = Conv1D(32, 10, activation='relu', padding = 'same')(x)
x = MaxPooling1D(4)(x)
x = Dropout(dropout_rate)(x)
x = Conv1D(32, 10, activation='relu', padding = 'same')(x)
x = Flatten()(x)
encoded = Dense(16, activation='tanh')(x)

x = Dense(125)(encoded)
x = Reshape((125,1))(x)
x = Conv1D(32, 10, activation='relu', padding = 'same')(x)
x = Dropout(dropout_rate)(x)
x = UpSampling1D(2)(x)
x = Conv1D(32, 10, activation='relu', padding = 'same')(x)
x = Dropout(dropout_rate)(x)
x = UpSampling1D(2)(x)
x = Conv1D(16, 10, activation='tanh', padding = 'same')(x)
decoded = Flatten()(x)
#decoded = Dense(8000, activation = 'tanh')(x)

autoencoder = Model(snd_in, decoded)
autoencoder.summary()
optim = Adam(lr = 0.0005)
autoencoder.compile(optimizer = optim, loss = 'mean_squared_error')
#autoencoder = load_model('aecd_snd')

#autoencoder.fit(file_in, file_in,
#                epochs = 50,
#                batch_size = 500,
#                shuffle = True,
#                validation_data = (eval_in,eval_in))
#autoencoder.save('aecd_snd')
#
#allin = []
#tmp = autoencoder.predict(eval_in)
#np.save('eval_out.npy',tmp)
del autoencoder
