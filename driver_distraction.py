import mediapipe as mp
import cv2
import numpy as np


class HeadPoseEstimator:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def get_head_pose(self, frame):
        image_height, image_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.mp_face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            landmarks, head_position, head_rotation, eye_positions, mouth_position, eye_aperture, eye_aperture_confidence = self.extract_landmarks(face_landmarks, image_width, image_height)
            return landmarks, head_position, head_rotation, eye_positions, mouth_position, eye_aperture, eye_aperture_confidence

        return None, None, None, None, None, None, None

    @staticmethod
    def extract_landmarks(face_landmarks, image_width, image_height):
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            z = landmark.z
            landmarks.append((x, y, z))

        model_points = np.array([
            landmarks[30],      # Nose tip
            landmarks[8],       # Chin
            landmarks[36],      # Left eye left corner
            landmarks[45],      # Right eye right corner
            landmarks[48],      # Left Mouth corner
            landmarks[54],      # Right mouth corner
            landmarks[2],       # Left eyebrow outer end
            landmarks[14],      # Right eyebrow outer end
            landmarks[33],      # Nose bottom
            landmarks[62],      # Upper lip center
            landmarks[66]       # Lower lip center
        ], dtype=float)

        vehicle_coordinate_system = np.array([
            [1, 0, 0],    # X axis
            [0, -1, 0],   # Y axis
            [0, 0, -1]    # Z axis
        ])

        vehicle_model_points = np.dot(model_points - np.array([0, 0, 0]), vehicle_coordinate_system.T)

        head_position = vehicle_model_points[0]
        head_rotation = vehicle_model_points[1]
        eye_positions = [vehicle_model_points[2], vehicle_model_points[3]]
        mouth_position = (vehicle_model_points[4] + vehicle_model_points[5]) / 2

        # Calculate eye aperture
        eye_aperture = np.linalg.norm(eye_positions[0] - eye_positions[1])
        eye_aperture_confidence = 0.8  # Placeholder value, you can modify this based on the actual confidence calculation

        return landmarks, head_position, head_rotation, eye_positions, mouth_position, eye_aperture, eye_aperture_confidence


head_pose_estimator = HeadPoseEstimator()


webcam = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    _, frame = webcam.read()

    # Estimate head pose and extract information
    landmarks, head_position, head_rotation, eye_positions, mouth_position, eye_aperture, eye_aperture_confidence = head_pose_estimator.get_head_pose(frame)

    if landmarks is not None:
        # Draw head pose info on the frame
        for (x, y, z) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Display head position and rotation
        cv2.putText(frame, f"Head Position: {head_position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Head Rotation: {head_rotation}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display eye positions
        cv2.putText(frame, f"Left Eye Position: {eye_positions[0]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Right Eye Position: {eye_positions[1]}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display mouth position
        cv2.putText(frame, f"Mouth Position: {mouth_position}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display eye aperture and confidence level
        cv2.putText(frame, f"Eye Aperture: {eye_aperture} mm", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow("Head Pose", frame)
        if cv2.waitKey(30) == 27:
            break


webcam.release()
cv2.destroyAllWindows()
