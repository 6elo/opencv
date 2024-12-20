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
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    image_height, image_width, _ = image.shape
    #to improve performance
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
###########################################################################################################################################################################
        ids0, ids1, ids2, ids3, ids4 = [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0] #quest
        ids5, ids6, ids7, ids8, ids9 = [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0] #quest
        ids10, ids11, ids12, ids13, ids14 = [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0] #quest
        ids15, ids16, ids17, ids18, ids19, ids20 = [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0], [0, 0,0] #quest
        x_max = 0
        y_max = 0
        x_min = image_width
        y_min = image_height
        #fist=100
        gesture="unknown gesture"
        countFin=0
        openedFing=[0,0,0,0,0]
###########################################################################################################################################################################
        for ids, landmrk in enumerate(hand_landmarks.landmark):
            cx = int(landmrk.x * image_width) # position x
            cy = int(landmrk.y * image_height) # position Y
            cz = int(landmrk.z)  # position z
            if cx > x_max:
                x_max = cx
            if cx < x_min:
                x_min = cx
            if cy > y_max:
                y_max = cy
            if cy < y_min:
                y_min = cy
############################################################################################################################################################################
#QUEST:
            #0 -create  variables/list/field for every landmarks (0-20)  (handmark_number=ids,position x, position y)
            if ids == 0:
              ids0 = [cx, cy, cz]
            elif ids == 1:
              ids1 = [cx, cy, cz]
            elif ids == 2:
              ids2 = [cx, cy, cz]
            elif ids == 3:
              ids3 = [cx, cy, cz]
            elif ids == 4:
              ids4 = [cx, cy, cz]
            elif ids == 5:
              ids5 = [cx, cy, cz]
            elif ids == 6:
              ids6 = [cx, cy, cz]
            elif ids == 7:
              ids7 = [cx, cy, cz]
            elif ids == 8:
              ids8 = [cx, cy, cz]
            elif ids == 9:
              ids9 = [cx, cy, cz]
            elif ids == 10:
              ids10 = [cx, cy, cz]
            elif ids == 11:
              ids11 = [cx, cy, cz]
            elif ids == 12:
              ids12 = [cx, cy, cz]
            elif ids == 13:
              ids13 = [cx, cy, cz]
            elif ids == 14:
              ids14 = [cx, cy, cz]
            elif ids == 15:
              ids15 = [cx, cy, cz]
            elif ids == 16:
              ids16 = [cx, cy, cz]
            elif ids == 17:
              ids17 = [cx, cy, cz]
            elif ids == 18:
              ids18 = [cx, cy, cz]
            elif ids == 19:
              ids19 = [cx, cy, cz]
            elif ids == 20:
              ids20 = [cx, cy, cz]

            #1 -calculate distance between two points
            def distance (p1 , p2):
              dis=abs(int(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 +(p1[2] - p2[2]) ** 2))) #alternative way to calculate distance of points in 3D world look math "sqrt" and "hypot"
              return dis
            
            #fist
            if distance(ids0,ids4)<distance(ids0,ids11) and distance(ids0,ids8)<distance(ids0,ids5) and distance(ids0,ids12)<distance(ids0,ids9) and distance(ids0,ids16)<distance(ids0,ids13) and distance(ids0,ids20)<distance(ids0,ids17):
                gesture="any distance fist"

            #number of fingers is opened + which one is opened
            if distance(ids4,ids0)>(distance(ids0,ids1)+distance(ids1,ids2)+distance(ids2,ids3)+distance(ids3,ids4)):
              countFin=countFin+1
              openedFing[0]="thumb"
            if distance(ids8,ids0)>(distance(ids0,ids5)+distance(ids5,ids6)+distance(ids6,ids7)+distance(ids7,ids8)):
              countFin=countFin+1
              openedFing[1]="index"
            if distance(ids12,ids0)>(distance(ids0,ids9)+distance(ids9,ids10)+distance(ids10,ids11)+distance(ids11,ids12)):
              countFin=countFin+1
              openedFing[2]="middle"
            if distance(ids16,ids0)>(distance(ids0,ids13)+distance(ids13,ids14)+distance(ids14,ids15)+distance(ids15,ids16)):
              countFin=countFin+1
              openedFing[3]="ring"
            if distance(ids20,ids0)>(distance(ids0,ids17)+distance(ids17,ids18)+distance(ids18,ids19)+distance(ids19,ids20)):
              countFin=countFin+1
              openedFing[4]="pinky"

########draw hand point, line from poin to point, write the lenth of line.... just exemple to remember
        cv2.circle(image, (ids8[0],ids8[1]), 10, (255, 0, 128), cv2.FILLED)
        #cv2.circle(image, (ids0[0],ids0[1]), fist, (255, 0, 128),3) #cirkle from wrist like a fist-big
        cv2.line(image, (ids8[0],ids8[1]), (ids4[0],ids4[1]), (255, 0, 128), 3)
       
        #need to flip only text
        image = cv2.flip(image, 1)
        
        cv2.putText(image, gesture, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(countFin), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 125), 2)
        cv2.putText(image, str(openedFing), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 125, 0), 2)

        
        image = cv2.flip(image, 1)
#########draw rectangle around the hand
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) #draw rectangle around hand.... just exemple

            #2 -recognize fist (wrist_position and position of EVERY finger TIP at some distance away from wrist) 
            #3 -calculate how many fingers is opened....and print it, finger tip is at some distance away from MPC
            #4 -recognize gesto and print its name (compared position of  landmarks and recognize gesto)
            #5 -make it work at any distance away from camera #chek distance from wrist to finger tip and compare it with lenth from wrist to tip    
            #6 -slice program for better reading, make separate gesture program


############################################################################################################################################################################
    # Flip the image horizontally for a mirror-view
    cv2.imshow('HandTracking', cv2.flip(image, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()