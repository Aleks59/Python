import tensorflow as tf
import numpy as np
import os
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense, Activation, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D, Flatten, Reshape
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


filename_ev = 'training_data_eval.npy'

filename = 'training_data_norm.npy'
filename1 = 'training_data_norm1.npy'
filename2 = 'training_data_norm2.npy'
filename3 = 'training_data_norm3.npy'

if os.path.isfile(filename):
    train_data = np.load(filename) 
    train_data1 = np.load(filename1)    
    train_data2 = np.load(filename2)
    train_data3 = np.load(filename3)
    
    eval_data = np.load(filename_ev)

tr_in = [] 
ev_in = []
for data in train_data:
    tr_in.append(data[0])
for data in train_data1:
    tr_in.append(data[0])
for data in train_data2:
    tr_in.append(data[0])
for data in train_data3:
    tr_in.append(data[0])    
    
for data in eval_data:
    ev_in.append(data[0])

train_in = np.array(tr_in)

eval_in = np.array(ev_in)

train_in = train_in.reshape([-1,44,32,1])
eval_in = eval_in.reshape([-1,44,32,1])

# Building the encoder
input_img = Input(shape=(44,32,1))

x = Conv2D(16, 3, activation = 'relu', padding = 'same')(input_img)
x = MaxPooling2D(2, padding = 'same')(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = MaxPooling2D(2, padding='same')(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = Flatten()(x)
encoded = Dense(64)(x)

x = Dense(16*8*11)(encoded)
x = Reshape((11,8,16))(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = UpSampling2D(2)(x)
#x = Conv2D(16, 3, activation='relu', padding='same')(x)
decoded = Conv2D(1, 3, activation = 'sigmoid', padding = 'same')(x)

autoencoder = Model(input_img, decoded)
optim = Adam(lr = 0.0005)
autoencoder.compile(optimizer = optim, loss = 'mean_squared_error')
#autoencoder = load_model('kerasAECDnoise')

autoencoder.fit(train_in, train_in,
                epochs = 100,
                batch_size = 512,
                shuffle = True,
                validation_data = (eval_in,eval_in))
autoencoder.save('aecd')

allin = []
tmp = autoencoder.predict(eval_in)
for data in tmp:
    img = data.reshape([44,32])
    allin.append(img)    
#    cv2.imshow('eval',img)
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        break
np.save('aecd.npy',allin)
#cv2.destroyAllWindows()
del autoencoder
K.clear_session()