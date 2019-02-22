# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 02:02:19 2018

@author: Александр
"""

import pygame
import math

def drawPend(x0,y0,x1,y1,angle):
    pygame.draw.line(screen,BLACK,[int(x0),int(y0)],[int(x1),int(y1)],3)
    pygame.draw.circle(screen,BLACK,[int(x1),int(y1)],9,3)

g = 1.0
m1 = 1.0
m2 = 1.0
l1 = 50.0
l2 = 50.0

an1_v = 0.0
an2_v = 0.0
an1_a = 0.0
an2_a = 0.0

size = (500,500)
done = False

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

cX = size[0]/2
cY = size[1]/2

pygame.init()
pygame.font.init()
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()
TNRfont = pygame.font.SysFont('Times New Roman', 15)

an1 = math.pi/2
an2 = 0.0

#=========================
while done==False:
    #cycle for work,
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    #general code part   
    
    num1 = -g*(2*m1+m2)*math.sin(an1)
    num2 = -m2*g*math.sin(an1-2*an2)
    num3 = -2*math.sin(an1-an2)
    num4 = m2*(an2_v*an2_v*l2+an1_v*an1_v*l1*math.cos(an1-an2))
    den = l1*(2*m1+m2-m2*math.cos(2*an1-2*an2))
    
    an1_a = (num1+num2+num3*num4)/den
    
    num1 = 2*math.sin(an1-an2)
    num2 = an1_v*an1_v*l1*(m1+m2)
    num3 = g*(m1+m2)*math.cos(an1)
    num4 = an2_v*an2_v*l2*m2*math.cos(an1-an2)
    den = l2*(2*m1+m2-m2*math.cos(2*an1-2*an2))
    
    an2_a = num1*(num2+num3+num4)/den
    
    an1_v +=an1_a
    an2_v +=an2_a
    
    an1_v=0.99*an1_v
    an2_v=0.99*an2_v
    
    an1 +=an1_v
    an2 +=an2_v 
    
    ken1 = m1*an1_v*an1_v/2
    ken2 = m2*an2_v*an2_v/2
    k1Text =TNRfont.render('Kinetic1: '+str(ken1),False,BLACK)
    k2Text =TNRfont.render('Kinetic2: '+str(ken2),False,BLACK)
    
    p1x = l1*math.sin(an1)
    p1y = l1*math.cos(an1)
    p2x = l2*math.sin(an2)
    p2y = l2*math.cos(an2)
    
    #set background
    screen.fill(WHITE)
    
    #drawing part
    drawPend(cX,cY,cX+p1x,cY+p1y,an1)
    drawPend(cX+p1x,cY+p1y,cX+p1x+p2x,cY+p1y+p2y,an2)
    screen.blit(k1Text,(10,10))
    screen.blit(k2Text,(10,20))
    #update screen
    pygame.display.flip()     
    
    #FPS
    clock.tick(60)
pygame.font.quit()
pygame.quit()