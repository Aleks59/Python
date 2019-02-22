import time

import cv2
import numpy as np
from grabscreen import grab_screen
import win32api
import win32con
import win32gui
import os
    
def keys_to_output():

    output = [0,0,0]
    
    if win32api.GetAsyncKeyState(ord('Z')):
        output[0] = -1
    if win32api.GetAsyncKeyState(ord('X')):
        output[0] = 1
    if win32api.GetAsyncKeyState(win32con.VK_LEFT):
        output[1] = -1
    if win32api.GetAsyncKeyState(win32con.VK_DOWN):
        output[2] = 1
    if win32api.GetAsyncKeyState(win32con.VK_RIGHT):
        output[1] = 1
    
    return output 

file_name = 'mul/fps.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh')
    training_data = []

def main():

    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)        
    start_time = time.time();
    while(True):
        tmp_time = time.time() - start_time  
        if 15*tmp_time > 1000:
            try:
                hwnd = win32gui.FindWindow(None, 'Not Tetris 2')
                find_region = win32gui.GetWindowRect(hwnd)
            except:
                find_region = (0,0,100,100)
            left,top,x2,y2 = find_region        
            #screen = grab_screen(region=(left+3,top+25,x2-196,y2-1)) 
            # 288, 435
            screen = grab_screen(region=(left+46,top+25,x2-196,y2-1))       
            #245, 435
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            # resize to something a bit more acceptable for a CNN
            #screen = cv2.resize(screen, (32, 55))
            screen = cv2.resize(screen, (24,44))            
    #        screen = cv2.resize(screen, (64,64))
            output = keys_to_output()
            #print(output)                  
            training_data.append([screen/255.0,output])
            cv2.imshow('OpenCV/fps', screen)
            #time4scr = time.time() - last_time
            #print("time for 1 screen {0}", time4scr)
    #        if len(training_data) % 500 == 0 :
    #            print(len(training_data))    
            start_time = time.time()        
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break        
    np.save(file_name,training_data)
main()
