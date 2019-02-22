import numpy as np

from keras.layers import Dropout, Reshape, Flatten
from keras.layers import Lambda, Dense, Input, Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation
from keras.models import Model, load_model

from keras import backend as K
import tensorflow as tf
sess = tf.Session()
K.set_session(sess)

# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.6
 
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))

# import files for train

file0 = np.load("inae/f3_0.npy")
file1 = np.load("inae/f3_1.npy")
file2 = np.load("inae/f3_2.npy")
file3 = np.load("inae/f3_3.npy")
#eval_in = np.load("eval_in.npy")

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

#global const
batch_size = 50
batch_shape = (batch_size, 8000)
latent_dim = 16
dropout_rate = 0.3
gamma = 0.5

#batch iterators
def gen_batch(x):
    n_batches = x.shape[0] // batch_size
    while(True):
        idxs = np.random.permutation(x.shape[0])
        x = x[idxs]
        for i in range(n_batches):
            yield x[batch_size*i: batch_size*(i+1)]
train_batches_it = gen_batch(file_in)
#test_batches_it = gen_batch(eval_in)

x_ = tf.placeholder(tf.float32, shape = (None, 8000), name = 'sound')
z_ = tf.placeholder(tf.float32, shape = (None, latent_dim), name ='z')

snd = Input(tensor=x_)
z = Input(tensor=z_)

with tf.variable_scope('encoder'):
    x = Reshape((8000,1))(snd)
    x = Conv1D(16, 10, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(64, 5, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(64, 2, activation='tanh', padding = 'same')(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(4)(x)
    h = Flatten()(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,stddev=1.0)
        return z_mean + K.exp(K.clip(z_log_var/2, -2, 2)) * epsilon
    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model([snd], [z_mean, z_log_var, l], name = 'Encoder')

with tf.variable_scope('decoder'):
    x = Dense(125)(z)
    x = Dropout(dropout_rate)(x)
    x = Reshape((125,1))(x)
    x = Conv1D(64, 3, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='tanh', padding = 'same')(x)
    decoded = Flatten()(x)
decoder = Model([z], decoded, name='Decoder')

with tf.variable_scope('discrim'):
    x = Reshape((8000, 1))(snd)
    x = Conv1D(16, 8, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(16, 4, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = MaxPooling1D(4)(x)
    x = Dropout(dropout_rate)(x)
    l = Conv1D(16, 2, padding = 'same')(x)
    x = Dropout(dropout_rate)(l)
    h = Flatten()(l)
    x = Dense(128)(h)
    
    d = Dense(1, activation = 'sigmoid')(x)
discrim = Model([snd], [d,l], name = 'Discriminator')

#graph
z_mean, z_log_var, encoded_snd = encoder([snd])

decoded_snd = decoder([encoded_snd])
decoded_z = decoder([z])

discr_snd,      discr_l_snd     = discrim([snd])
discr_dec_snd,  discr_l_dec_snd = discrim([decoded_snd])
discr_dec_z,    discr_l_dec_z   = discrim([decoded_z])

cvae_model = Model([snd], decoder([encoded_snd]), name='cvae')
cvae = cvae_model([snd])

#basic loss
L_prior = -0.5*tf.reduce_sum(1. + tf.clip_by_value(z_log_var, -2, 2) - 
                             tf.square(z_mean)-tf.exp(tf.clip_by_value(z_log_var,-2,2)))/8000
log_dis_snd     = tf.log(discr_snd + 1e-10)
log_dis_dec_z   = tf.log(1. - discr_dec_z +1e-10)

#log_dis_dec_snd = tf.log(1. - discr_dec_snd + 1e-10)
log_dis_dec_snd = -tf.log(discr_dec_snd + 1e-10)

L_GAN = -1/4*tf.reduce_sum(log_dis_snd+2*log_dis_dec_z+log_dis_dec_snd)/8000

L_dis_llike = tf.reduce_sum(tf.square(discr_l_snd - discr_l_dec_snd))/8000

#encoder, decoder and discrim losses
L_enc = L_dis_llike + L_prior
L_dec = gamma * L_dis_llike - L_GAN
L_dis = L_GAN

optimizer_enc = tf.train.AdamOptimizer(0.0005)
optimizer_dec = tf.train.AdamOptimizer(0.0003)
optimizer_dis = tf.train.AdamOptimizer(0.0005)

encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim")

step_enc = optimizer_enc.minimize(L_enc, var_list=encoder_vars)
step_dec = optimizer_dec.minimize(L_dec, var_list=decoder_vars)
step_dis = optimizer_dis.minimize(L_dis, var_list=discrim_vars)

def step(sound, zp):
    l_prior, dec_snd, l_dis_llike, l_gan, _, _ = sess.run([L_prior, decoded_z,
                                                           L_dis_llike, L_GAN, step_enc,
                                                           step_dec],
    feed_dict = {z:zp, snd:sound,K.learning_phase():1})
    return l_prior, dec_snd, l_dis_llike, l_gan

def step_d(sound, zp):
    l_gan, _ = sess.run([L_GAN, step_dis], feed_dict = {z:zp, 
                        snd:sound, K.learning_phase():1})
    return l_gan

#learning
sess.run(tf.global_variables_initializer())

nb_step = 3

saver = tf.train.Saver()
saver.save(sess, './model')
batches_per_period = 3
for i in range(15000):
    print('.', end ='')
    
    for j in range(nb_step):
        b0 = next(train_batches_it)
        zp = np.random.randn(batch_size, latent_dim)
        l_g = step_d(b0, zp)
        if l_g < 1.0:
            break
        
    for j in range(nb_step):
        l_p, zx, l_d, l_g = step(b0, zp)
        if l_g > 0.4:
            break
        b0 = next(train_batches_it)
        zp = np.random.randn(batch_size, latent_dim)
        
    if not i % batches_per_period:
        print(i, l_p, l_d, l_g)

del file_in