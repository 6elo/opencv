# Include some examples from MediaPipe library to draw lines, circles, rectangles, text, and to get landmarks of hand
# This code is a bit optimized but a bit worse to read for kids

# Work only with Python version 3.11 and lower
# To run this code you need to install mediapipe library, opencv-python, and math
# pip install mediapipe
# pip install opencv-python
# math is a standard library in python
# press "Q" to exit program and close camera

# QUEST:
# 0 - Create variables/list/field for every landmark (0-20) (handmark_number=ids, position x, position y, position z)
# 1 - Calculate distance between two points
# 2 - Recognize fist (wrist_position and position of EVERY finger TIP at some distance away from wrist)
# 3 - Calculate how many fingers are opened and print it, finger tip is at some distance away from wrist
# 4 - Recognize gesture and print its name (compare position of landmarks and recognize gesture)
# 5 - Make it work at any distance away from camera (check distance from wrist to finger tip and compare it with length from wrist to tip)
# 6 - Use two hands at the same time
# 7 - Make a history of last 10 frames change
# 8 - Make a mouse with gesture
# 9 - Make zoom in and out with gesture
# 10 - Make zoom in and out animated gesture like a touchpad
# 11 - Make hidden keyboard with gesture

import cv2
import mediapipe as mp
import math
import time
import copy
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

# Variables
last_update_time = time.time()  # Initialize once outside the loop to get the current time when the program starts
idsMass = [[[0, 0, 0] for _ in range(21)], [[0, 0, 0] for _ in range(21)]]  # 0 - left hand, 1 - right hand
idsMass_history = [copy.deepcopy(idsMass) for _ in range(9)]  # History of the last 10 frames change
prev_x, prev_y = None, None
prev_frame_time = 0

# Functions
def distance(hand1, landmark1, hand2, landmark2):
    dis = abs(int(math.sqrt(
        (idsMass[hand1][landmark1][0] - idsMass[hand2][landmark2][0]) ** 2 +
        (idsMass[hand1][landmark1][1] - idsMass[hand2][landmark2][1]) ** 2 +
        (idsMass[hand1][landmark1][2] - idsMass[hand2][landmark2][2]) ** 2
    )))
    return dis

def fps_calculate():
    global prev_frame_time
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    return fps

def history_update():
    current_time = time.time()
    global last_update_time
    if (current_time - last_update_time) >= 0.1:
        idsMass_history.pop(0)
        idsMass_history.append(copy.deepcopy(idsMass))
        last_update_time = current_time

def history_points(image):
    for hand_history in idsMass_history:
        for hand in hand_history:
            for fingertip in [4, 8, 12, 16, 20]:  # Assuming these are the indices for the fingertips
                cv2.circle(image, (hand[fingertip][0], hand[fingertip][1]), 5, (255, 0, 128), cv2.FILLED)

def control_mouse_with_gesture(point):
    global prev_x, prev_y
    screen_width, screen_height = pyautogui.size()
    index_finger_tip = point  # hand landmarks for control mouse

    # Calculate the position of the index finger tip
    x = int((index_finger_tip[0] * screen_width / image.shape[1]) * (-1))
    y = int(index_finger_tip[1] * screen_height / image.shape[0])

    # Move the mouse cursor if the index finger is straight
    if distance(0, 8, 0, 6) > distance(0, 6, 0, 5):
        if prev_x is not None and prev_y is not None:
            dx = x - prev_x
            dy = y - prev_y
            if abs(dx) > 15 or abs(dy) > 15:
                pyautogui.moveRel(dx, dy)
        prev_x, prev_y = x, y

def recognize_fist():
    gesture = "unknown gesture"
    if (distance(0, 0, 0, 4) < distance(0, 0, 0, 11) and
        distance(0, 0, 0, 8) < distance(0, 0, 0, 5) and
        distance(0, 0, 0, 12) < distance(0, 0, 0, 9) and
        distance(0, 0, 0, 16) < distance(0, 0, 0, 13) and
        distance(0, 0, 0, 20) < distance(0, 0, 0, 17)):
        gesture = "fist"
    return gesture

def count_open_fingers():
    countFin = 0
    openedFing = ["no", "no", "no", "no", "no"]

    if distance(0, 4, 0, 0) >= (distance(0, 0, 0, 1) + distance(0, 1, 0, 2) + distance(0, 2, 0, 3)):
        countFin += 1
        openedFing[0] = "thumb"

    if distance(0, 8, 0, 0) >= (distance(0, 0, 0, 5) + distance(0, 5, 0, 6) + distance(0, 6, 0, 7)):
        countFin += 1
        openedFing[1] = "index"

    if distance(0, 12, 0, 0) >= (distance(0, 0, 0, 9) + distance(0, 9, 0, 10) + distance(0, 10, 0, 11)):
        countFin += 1
        openedFing[2] = "middle"

    if distance(0, 16, 0, 0) >= (distance(0, 0, 0, 13) + distance(0, 13, 0, 14) + distance(0, 14, 0, 15)):
        countFin += 1
        openedFing[3] = "ring"

    if distance(0, 20, 0, 0) >= (distance(0, 0, 0, 17) + distance(0, 17, 0, 18) + distance(0, 18, 0, 19)):
        countFin += 1
        openedFing[4] = "pinky"

    return countFin, openedFing

# Main loop to process video frames
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    max_num_hands=2,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_height, image_width, _ = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        x_max, y_max = 0, 0
        x_min, y_min = image_width, image_height

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_info.classification[0].label  # 'Left' or 'Right'
                hand_type = 1 if hand_label == "Left" else 0  # 1 - right hand, 0 - left hand

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    cx = int(landmrk.x * image_width)
                    cy = int(landmrk.y * image_height)
                    cz = int(landmrk.z)
                    idsMass[hand_type][ids] = [cx, cy, cz]

                    if cx > x_max:
                        x_max = cx
                    if cx < x_min:
                        x_min = cx
                    if cy > y_max:
                        y_max = cy
                    if cy < y_min:
                        y_min = cy

        gesture = recognize_fist()
        countFin, openedFing = count_open_fingers()  
        history_update()
        history_points(image)
        fps = fps_calculate()
        control_mouse_with_gesture(idsMass[0][8])

        # Draw text and other elements on the image
        image = cv2.flip(image, 1)
        cv2.putText(image, gesture, (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, ("right" if hand_type == 1 else "left"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, str(countFin), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 125, 125), 2)
        cv2.putText(image, str(openedFing), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (125, 125, 0), 2)
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        image = cv2.flip(image, 1)

        # Show the image
        cv2.imshow('HandTracking', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()