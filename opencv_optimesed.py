
#Include some exemples from mediaPipe library to draw lines,circles, rectangles, text, and to get landmarks of hand
#This code is a bit optiamised but a bit worse to read for kids

#work only with Python version 3.11 and lower
#to run this code you need to install mediapipe library, opencv-python, and math
#pip install mediapipe
#pip install opencv-python
#math is a standard library in python
#press "q" to exit program and close camera


#QUEST: 
#0 -create  variables/list/field for every landmarks (0-20)  (handmark_number=ids,position x, position y, position z)
#1 -calculate distance between two points
#2 -recognize fist (wrist_position and position of EVERY finger TIP at some distance away from wrist) 
#3 -calculate how many fingers is opened....and print it, finger tip is at some distance away from MPC
#4 -recognize gesto and print its name (compared position of  landmarks and recognize gesto)
#5 -make it work at any distance away from camera #chek distance from wrist to finger tip and compare it with lenth from wrist to tip    
#6 -slice program for better reading, make separate gesture program
#7 -make virtual mouse with gesto
#8 -use some gesto for macros/system control/
#9 -make animated gesto
#10 -use more hands
#11 -make hiden keyboard with gesto

import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=2,
    min_tracking_confidence=0.5) as hands:
#start of our cycle
  while cap.isOpened():
    success, image = cap.read()
    image_height, image_width, _ = image.shape
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks,results.multi_handedness):
            hand_label = hand_info.classification[0].label  # 'Left' or 'Right'

            #change hand label to right or left, there fliped hands in mirror view so we need to change it here for better reading, if u dont use 
            #mirror view u dont need to change it and can use directly hand_label
            hand_type = "right" if hand_label == "Left" else "left" 
            
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style())
        x_max = 0
        y_max = 0
        x_min = image_width
        y_min = image_height
       

        #create  variables/list/field for every landmarks (0-20)  (handmark_number=ids,position x, position y, position z)
        idsMass=[[0, 0, 0]]*21

        #wer used this part of code to get landmarks of hand, make them as a list and store them in idsMass
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            cx = int(landmrk.x * image_width) 
            cy = int(landmrk.y * image_height)
            cz = int(landmrk.z)
            idsMass[ids] = [cx, cy, cz]
            
            #if u dont need to draw rectangle around hand, u can delete this part of code
            #find the biggest and smallest X ... for rectangel for exemple 
            if cx > x_max:
                x_max = cx
            if cx < x_min:
                x_min = cx
            #find the biggest and smallest Y ... for rectangel for exemple
            if cy > y_max:
                y_max = cy
            if cy < y_min:
                y_min = cy

            gesture="unknown gesture"
            countFin=0
            openedFing=["no","no","no","no","no"]

            #fast function to calculate distance between two points
            def distance(p1,p2):
                dis=abs(int(math.sqrt(((idsMass[p1][0]) - (idsMass[p2][0])) ** 2 + ((idsMass[p1][1]) - (idsMass[p2][1])) ** 2 +((idsMass[p1][2]) - (idsMass[p2][2])) ** 2))) #alternative way to calculate distance of points in 3D world look math "sqrt" and "hypot"
                return dis
            
            #Ur first gesture is fist
            if distance(0,4) < distance(0,11) and distance(0,8) < distance(0,5) and distance(0,12) < distance(0,9) and distance(0,16) < distance(0,13) and distance(0,20) < distance(0,17):
                gesture="any distance fist"
            
            #number of fingers is opened + which one is opened
            if distance(4,0) >= (distance(0,1) + distance(1,2) + distance(2,3)):
                countFin=countFin+1
                openedFing[0]="thumb"

            if distance(8,0) >= (distance(0,5)+distance(5,6)+distance(6,7)):
               countFin=countFin+1
               openedFing[1]="index"

            if distance(12,0) >= (distance(0,9)+distance(9,10)+distance(10,11)):
               countFin=countFin+1
               openedFing[2]="middle"

            if distance(16,0) >= (distance(0,13)+distance(13,14)+distance(14,15)):
               countFin=countFin+1
               openedFing[3]="ring"

            if distance(20,0) >= (distance(0,17)+distance(17,18)+distance(18,19)):
               countFin=countFin+1
               openedFing[4]="pinky"
            
        #to draw some shapes on image
        # #to draw a circle on "image" with center in "ids8" and "radius 10", "color (255, 0, 128)" and "filled circle, use number for thickness to draw circle    
        cv2.circle(image, (idsMass[8][0],idsMass[8][1]), 10, (255, 0, 128), cv2.FILLED)

        #to draw a line between idsmass[8] and idsmass[4] with color (255, 0, 128) and thickness 3
        cv2.line(image, (idsMass[8][0],idsMass[8][1]), (idsMass[4][0],idsMass[4][1]), (255, 0, 128), 3) 

        #to draw a rectangle around hand(smalest and highest points) on image with color (0, 255, 0) and thickness 2
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        #need to flip only text, write any text inside of two flips
        image = cv2.flip(image, 1) 
        #write text on image with text "gesture" on position (80,100) with font "cv2.FONT_HERSHEY_SIMPLEX", size 1, color (0, 0, 255), thickness 2
        cv2.putText(image, gesture, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
        cv2.putText(image, hand_type, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(countFin), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 125), 2)
        cv2.putText(image, str(openedFing), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 125, 0), 2)
        image = cv2.flip(image, 1)

    # Flip the image horizontally for a mirror-view
    cv2.imshow('HandTracking', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()