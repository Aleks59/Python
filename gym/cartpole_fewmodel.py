# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 01:10:45 2018

@author: Александр
"""
import tensorflow as tf
import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 1e-3
env = gym.make("CartPole-v1")
env.reset()
goal_steps = 500
tf.reset_default_graph()
def neural_network_model(input_size, model_num):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='logs1n/'+str(model_num))

    return model

def train_model(training_data, model_num, model=False):

    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size = len(X[0]), model_num = model_num)
    
    model.fit({'input': X}, {'targets': y}, n_epoch=1, snapshot_step=500, show_metric=True)
    return model

training_data = np.load('saved.npy')
models = []
all_score = []
for each_m in range(10):    
    models.append(train_model(training_data,each_m))
    scores = []
    choices = []
    for each_game in range(100):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            #env.render()
    
            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(models[each_m].predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
    
            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break
    
        scores.append(score)        
    tf.reset_default_graph()
    all_score.append(sum(scores)/len(scores))
    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
print(all_score)
env.close()