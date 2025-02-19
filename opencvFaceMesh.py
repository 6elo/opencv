import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
idsMass = [[0, 0, 0] for _ in range(478)] # Initialize idsMass with 478 elements

# Calibration variables
calibration_points = []
calibrated = False
screen_width, screen_height = pyautogui.size()

def default_face_mesh_tesselation_style(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

def default_face_mesh_contours_style(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

def default_face_mesh_iris_connections_style(image, face_landmarks):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
def calibrate_iris_positions():
    global calibration_points, calibrated
    if len(calibration_points) == 0:
        print("Look at the top-left corner of the screen and press 'c'")
    elif len(calibration_points) == 1:
        print("Look at the top-right corner of the screen and press 'c'")
    elif len(calibration_points) == 2:
        print("Look at the bottom-left corner of the screen and press 'c'")
    elif len(calibration_points) == 3:
        print("Look at the bottom-right corner of the screen and press 'c'")
    elif len(calibration_points) == 4:
        calibrated = True
        print("Calibration complete!")

def mouse_controle():
    global calibrated
    if not calibrated:
        return
    # Use the average position of both irises for mouse control
    left_iris_x = (idsMass[468][0] + idsMass[469][0] + idsMass[470][0] + idsMass[471][0]) / 4
    left_iris_y = (idsMass[468][1] + idsMass[469][1] + idsMass[470][1] + idsMass[471][1]) / 4
    right_iris_x = (idsMass[472][0] + idsMass[473][0] + idsMass[474][0] + idsMass[475][0]) / 4
    right_iris_y = (idsMass[472][1] + idsMass[473][1] + idsMass[474][1] + idsMass[475][1]) / 4

    avg_iris_x = (left_iris_x + right_iris_x) / 2
    avg_iris_y = (left_iris_y + right_iris_y) / 2

    # Map the iris positions to screen coordinates
    screen_x = int(avg_iris_x * screen_width / image_width)
    screen_y = int(avg_iris_y * screen_height / image_height)

    pyautogui.moveTo(screen_x, screen_y)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_height, image_width, _ = image.shape
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #default_face_mesh_tesselation_style(image, face_landmarks)
                #default_face_mesh_contours_style(image, face_landmarks)
                #default_face_mesh_iris_connections_style(image, face_landmarks)

                # Process landmarks for each face
                for ids, landmrk in enumerate(face_landmarks.landmark):
                    cx = int(landmrk.x * image_width)
                    cy = int(landmrk.y * image_height)
                    cz = int(landmrk.z * 1000)  # Scale z for better visualization
                    idsMass[ids] = [cx, cy, cz]

        mouse_controle()

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):        
            break
        elif cv2.waitKey(1) & 0xFF == ord('c') or cv2.waitKey(1) & 0xFF == ord('C'):
            calibration_points.append((idsMass[468][0], idsMass[468][1]))
            calibrate_iris_positions()
cap.release()
cv2.destroyAllWindows()