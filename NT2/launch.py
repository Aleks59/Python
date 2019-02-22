import tensorflow as tf
from keras.models import load_model
import win32gui
import time
import keras
import gc
from grabscreen import grab_screen
import cv2
from pynput.keyboard import Key, Controller

model = load_model('kerasAECD2')

try:
    hwnd = win32gui.FindWindow(None, 'Not Tetris 2')
    find_region = win32gui.GetWindowRect(hwnd)
except:
    find_region = (0,0,100,100)
left,top,x2,y2 = find_region


for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
work = True
while work:
    image = grab_screen(region=(left+3,top+25,x2-196,y2-1))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('cd', image)    
    image = cv2.resize(image,(32,44))   
    image = image/255.0
    image = image.reshape([1,44,32,1])
    choise = model.predict(image)
    print(choise)    
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()


del model
keras.backend.clear_session()
tf.reset_default_graph()
gc.collect()