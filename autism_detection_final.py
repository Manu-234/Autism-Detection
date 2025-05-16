import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Set up webcam
cap = cv2.VideoCapture(0)

# Buffers to store previous hand positions
right_hand_history = deque(maxlen=20)
left_hand_history = deque(maxlen=20)

# Eye contact counter and threshold
eye_contact_counter = 0
eye_contact_threshold = 15  # How many consecutive frames indicate good eye contact

print("Running... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for mirror-like view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get pose estimation and face mesh
    pose_result = pose.process(rgb)
    face_result = face_mesh.process(rgb)

    h, w, _ = frame.shape

    # Hand Flapping Detection
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = pose_result.pose_landmarks.landmark

        # Get wrist landmarks and pixel coordinates
        right_wrist = landmarks[16]
        left_wrist = landmarks[15]

        rw_x, rw_y = int(right_wrist.x * w), int(right_wrist.y * h)
        lw_x, lw_y = int(left_wrist.x * w), int(left_wrist.y * h)

        # Append to history only if hand is clearly visible
        if right_wrist.visibility > 0.3:
            right_hand_history.append((rw_x, rw_y))

        if left_wrist.visibility > 0.3:
            left_hand_history.append((lw_x, lw_y))

        # Function to detect repetitive motion
        def detect_repetitive_motion(history, name):
            if len(history) < 10:
                return

            xs = [p[0] for p in history]
            ys = [p[1] for p in history]

            movement_x = np.std(xs)
            movement_y = np.std(ys)

            # Detect repetitive motion within reasonable range
            if 20 < movement_x < 60 and movement_y < 60:
                cv2.putText(frame, f"Repetitive {name} hand motion detected!",
                            (10, 40 if name == "Right" else 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Call with swapped labels due to flipped image
        detect_repetitive_motion(right_hand_history, "Right")
        detect_repetitive_motion(left_hand_history, "Left")

        # Draw wrist points
        if right_wrist.visibility > 0.3:
            cv2.circle(frame, (rw_x, rw_y), 8, (0, 255, 0), -1)
        if left_wrist.visibility > 0.3:
            cv2.circle(frame, (lw_x, lw_y), 8, (255, 0, 0), -1)

    # Eye Contact Detection
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            # Iris center approx
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]

            # Eye corners
            left_eye_outer = face_landmarks.landmark[33]
            left_eye_inner = face_landmarks.landmark[133]
            right_eye_outer = face_landmarks.landmark[362]
            right_eye_inner = face_landmarks.landmark[263]

            def to_pixel(lm):
                return int(lm.x * w), int(lm.y * h)

            def is_eye_contact(iris, inner, outer, w, h):
                iris_x, iris_y = int(iris.x * w), int(iris.y * h)
                inner_x, inner_y = int(inner.x * w), int(inner.y * h)
                outer_x, outer_y = int(outer.x * w), int(outer.y * h)

                eye_center_x = int((inner_x + outer_x) / 2)
                eye_center_y = int((inner_y + outer_y) / 2)

                eye_width = np.linalg.norm([outer_x - inner_x, outer_y - inner_y])
                distance = np.linalg.norm([iris_x - eye_center_x, iris_y - eye_center_y])

                return distance < 0.3 * eye_width

            left_centered = is_eye_contact(left_iris, left_eye_inner, left_eye_outer, w, h)
            right_centered = is_eye_contact(right_iris, right_eye_inner, right_eye_outer, w, h)

            if left_centered and right_centered:
                eye_contact_counter += 1
            else:
                eye_contact_counter = max(eye_contact_counter - 1, 0)

            if eye_contact_counter >= eye_contact_threshold:
                text = "Eye contact detected"
                color = (0, 255, 0)
            else:
                text = "No eye contact"
                color = (0, 0, 255)

            cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show output
    cv2.imshow("Autism Detection Combined", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
