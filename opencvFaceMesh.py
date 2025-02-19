import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
idsMass = [[0, 0, 0] for _ in range(478)] # Initialize idsMass with 478 elements

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



        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('Q'):        
            break

cap.release()
cv2.destroyAllWindows()