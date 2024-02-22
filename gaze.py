import mediapipe as mp
import cv2
import numpy as np
from math import acos, degrees
import os, csv

webcam = False
video = r"C:\Users\colonel\Desktop\meas\assets\videos\Isadora.mp4"

horizontal_scale_factor = .25
vertical_scale_factor = 0.05

vertical_correction = 3.8
horizontal_correction = -0.5

vector_scale = 100
smooth_factor = 2
vector_depth = -1.2

headers = ['Convergence_X', 'Convergence_Y']
csv_file = r"C:\Users\colonel\Desktop\gaze.csv"
if not os.path.exists(csv_file) or os.path.getsize('gaze.csv') == 0:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

class ScalarSmoothingFilter:
    def __init__(self, window_size=smooth_factor):
        self.window_size = window_size
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)

class SmoothingFilter:
    def __init__(self, window_size=smooth_factor):
        self.window_size = window_size
        self.values = []

    def update(self, new_value):
        self.values.append(new_value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

        # Initialization of sums for each vector component
        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        for value in self.values:
            sum_x += value[0]
            sum_y += value[1]
            sum_z += value[2]

        # Calculating the average for each component
        average_x = sum_x / len(self.values)
        average_y = sum_y / len(self.values)
        average_z = sum_z / len(self.values)

        # Return the average vector
        return (average_x, average_y, average_z)

yaw_filter = ScalarSmoothingFilter()
pitch_filter = ScalarSmoothingFilter()
roll_filter = ScalarSmoothingFilter()
corrected_gaze_filter = SmoothingFilter(window_size=5)
length_filter = ScalarSmoothingFilter(window_size=5)

def quadratic_adjustment(value):
    adjustment_factor = 0.5
    adjusted_value = (value ** 2) * adjustment_factor
    if value < 0:
        adjusted_value = -adjusted_value
    return adjusted_value
def draw_lines_between_points(image, face_landmarks, pairs, color=(0, 0, 255), thickness=1):
    for pair in pairs:
        point1 = face_landmarks.landmark[pair[0]]
        point2 = face_landmarks.landmark[pair[1]]
        x1, y1 = int(point1.x * image.shape[1]), int(point1.y * image.shape[0])
        x2, y2 = int(point2.x * image.shape[1]), int(point2.y * image.shape[0])
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def draw_corrected_gaze_vector(image, eye_center, corrected_gaze_vector, scale=100, color=(0, 255, 250), thickness=2):

    z_offset = vector_depth
    eye_center_shifted = (eye_center[0], eye_center[1], eye_center[2] + z_offset)

    # Coordonnées du point de fin du vecteur
    end_point = (eye_center[0] + corrected_gaze_vector[0] * scale,
                 eye_center[1] + corrected_gaze_vector[1] * scale,
                 eye_center[2] + corrected_gaze_vector[2] * scale)

    # Dessiner le vecteur
    cv2.arrowedLine(image, (int(eye_center_shifted[0]), int(eye_center_shifted[1])),(int(end_point[0]), int(end_point[1])), color, thickness)

def correct_gaze_vector(yaw, pitch, roll, gaze_vector):
    # Convert rotation angles to radians
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    roll_rad = np.radians(roll)
    # Create rotation matrices
    R_yaw = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                      [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                      [0, 0, 1]])
    R_pitch = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                        [0, 1, 0],
                        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll_rad), -np.sin(roll_rad)],
                       [0, np.sin(roll_rad), np.cos(roll_rad)]])
    # Combine rotation matrices
    R = np.dot(R_roll, np.dot(R_pitch, R_yaw))
    # Measure original length of the gaze vector
    original_length = np.linalg.norm(gaze_vector)
    # Normalize the gaze vector if it's not a zero vector
    if original_length > 0:
        normalized_gaze_vector = gaze_vector / original_length
    else:
        normalized_gaze_vector = gaze_vector
    # Apply the rotation to the normalized gaze vector
    corrected_gaze_vector = np.dot(R, normalized_gaze_vector)
    # Return both the corrected gaze vector and the original length
    return corrected_gaze_vector, original_length

def estimate_head_rotation(face_landmarks, image):
    height, width, _ = image.shape
    landmarks = [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark]

    # Extracting points of interest from converted face landmarks
    eye1 = landmarks[226]
    eye2 = landmarks[446]
    nose = landmarks[1]
    chin = landmarks[152]

    # Calculating vectors for yaw, pitch, and roll
    a = (eye1[0] - eye2[0], eye1[1] - eye2[1])
    b = (nose[0] - eye2[0], nose[1] - eye2[1])
    c = (chin[0] - nose[0], chin[1] - nose[1])

    # Calculating Yaw (using vector between eyes and nose)
    cos_theta = (a[0] * b[0] + a[1] * b[1]) / (np.sqrt(a[0]**2 + a[1]**2) * np.sqrt(b[0]**2 + b[1]**2))
    yaw = degrees(acos(cos_theta))

    # Calculating Pitch (using vector between nose and chin)
    cos_phi = (c[0] * b[0] + c[1] * b[1]) / (np.sqrt(c[0]**2 + c[1]**2) * np.sqrt(b[0]**2 + b[1]**2))
    pitch = degrees(acos(cos_phi))

    # Calculating Roll (using vector between eyes)
    roll = degrees(np.arccos(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))))
    if np.cross(a, c) > 0:
        roll *= -1
    if roll < 0:
        roll += 360
    yaw -= 27
    pitch -= 62
    roll -= 90
    return yaw, pitch, roll


# Initializing MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initializing video capture.
if webcam :
    cap = cv2.VideoCapture(0)
else :
    cap = cv2.VideoCapture(video)

# Dictionary to store landmark coordinates
eye_coords = {}


left_eye_pairs = [(226, 244), (247, 233), (30, 232), (29, 231), (27, 230), (28, 229), (56, 228), (33, 133)]
left_pupil_pairs = [(470, 472), (471, 469)]
right_eye_pairs = [(463, 446), (414, 261), (286, 448), (258, 449), (257, 450), (259, 451), (260, 452), (467, 453)]
right_pupil_pairs = [(475, 477), (476, 474)]

temp_gaze = (0, 0, 0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = face_mesh.process(image_rgb)
    image_rgb.flags.writeable = True

    image_center = (image.shape[1] // 2, image.shape[0] // 2)

    # Cross
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    color = (255, 255, 255)
    thickness = 1
    cv2.line(image, (center_x, 0), (center_x, height), color, thickness)
    cv2.line(image, (0, center_y), (width, center_y), color, thickness)

    # Update processing in the main loop
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_points, right_eye_points = [], []
            left_pupil_points, right_pupil_points = [], []



            ############### LEFT

            for a, b in left_eye_pairs:
                pa = (face_landmarks.landmark[a].x * image.shape[1],
                      face_landmarks.landmark[a].y * image.shape[0],
                      face_landmarks.landmark[a].z)
                pb = (face_landmarks.landmark[b].x * image.shape[1],
                      face_landmarks.landmark[b].y * image.shape[0],
                      face_landmarks.landmark[b].z)
                left_eye_points.append(((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2))

            for a, b in left_pupil_pairs:
                pa = (
                    int(face_landmarks.landmark[a].x * image.shape[1]),
                    int(face_landmarks.landmark[a].y * image.shape[0]),
                    face_landmarks.landmark[a].z)
                pb = (
                    int(face_landmarks.landmark[b].x * image.shape[1]),
                    int(face_landmarks.landmark[b].y * image.shape[0]),
                    face_landmarks.landmark[b].z)
                left_pupil_points.append(
                    ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2))

            if left_eye_points:
                left_eye_center = (sum(x for x, y, z in left_eye_points) / len(left_eye_points),
                              sum(y for x, y, z in left_eye_points) / len(left_eye_points),
                              sum(z for x, y, z in left_eye_points) / len(left_eye_points))

            if left_pupil_points:
                left_pupil_center = (sum(x for x, y, z in left_pupil_points) / len(left_pupil_points),
                                sum(y for x, y, z in left_pupil_points) / len(left_pupil_points),
                                sum(z for x, y, z in left_pupil_points) / len( left_pupil_points))

            left_gaze_vector = (left_pupil_center[0] - left_eye_center[0],
                              left_pupil_center[1] - left_eye_center[1],
                              left_pupil_center[2] - left_eye_center[2])

            ############# RIGHT

            for a, b in right_eye_pairs:
                pa = (face_landmarks.landmark[a].x * image.shape[1],
                      face_landmarks.landmark[a].y * image.shape[0],
                      face_landmarks.landmark[a].z)
                pb = (face_landmarks.landmark[b].x * image.shape[1],
                      face_landmarks.landmark[b].y * image.shape[0],
                      face_landmarks.landmark[b].z)
                right_eye_points.append(((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2))

            for a, b in right_pupil_pairs:
                pa = (
                    int(face_landmarks.landmark[a].x * image.shape[1]),
                    int(face_landmarks.landmark[a].y * image.shape[0]),
                    face_landmarks.landmark[a].z)
                pb = (
                    int(face_landmarks.landmark[b].x * image.shape[1]),
                    int(face_landmarks.landmark[b].y * image.shape[0]),
                    face_landmarks.landmark[b].z)
                right_pupil_points.append(
                    ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2))

            if right_eye_points:
                right_eye_center = (sum(x for x, y, z in right_eye_points) / len(right_eye_points),
                              sum(y for x, y, z in right_eye_points) / len(right_eye_points),
                              sum(z for x, y, z in right_eye_points) / len(
                                  right_eye_points))

            if right_pupil_points:
                right_pupil_center = (sum(x for x, y, z in right_pupil_points) / len(right_pupil_points),
                                sum(y for x, y, z in right_pupil_points) / len(right_pupil_points),
                                sum(z for x, y, z in right_pupil_points) / len( right_pupil_points))

            right_gaze_vector = (right_pupil_center[0] - right_eye_center[0],
                              right_pupil_center[1] - right_eye_center[1],
                              right_pupil_center[2] - right_eye_center[2])

            cv2.line(image, (int(left_eye_center[0]), int(left_eye_center[1])),
                     (int(left_eye_center[0] + left_gaze_vector[0]), int(left_eye_center[1] + left_gaze_vector[1])), (0, 0, 255), 1)

            cv2.line(image, (int(right_eye_center[0]), int(right_eye_center[1])),
                     (int(right_eye_center[0] + right_gaze_vector[0]), int(right_eye_center[1] + right_gaze_vector[1])), (0, 0, 255), 1)

            # Blink
            blink_threshold = 35
            right_eye_blink_distance = face_landmarks.landmark[145].y * image.shape[0] - face_landmarks.landmark[
                159].y * image.shape[0]

            if right_eye_blink_distance > blink_threshold:
                yaw, pitch, roll = estimate_head_rotation(face_landmarks, image)

                smoothed_yaw = yaw_filter.update(yaw)
                smoothed_pitch = pitch_filter.update(pitch)
                smoothed_roll = roll_filter.update(roll)

                left_corrected_gaze_vector, original_left_length = correct_gaze_vector(smoothed_yaw, smoothed_pitch, smoothed_roll, left_gaze_vector)
                right_corrected_gaze_vector, original_right_length = correct_gaze_vector(smoothed_yaw, smoothed_pitch, smoothed_roll, right_gaze_vector)

                original_left_length = length_filter.update(original_left_length)
                original_right_length = length_filter.update(original_right_length)

                left_corrected_gaze_vector *= original_left_length
                right_corrected_gaze_vector *= original_right_length

                left_corrected_gaze_vector[1] += vertical_correction
                right_corrected_gaze_vector[1] += vertical_correction
                left_corrected_gaze_vector[0] += horizontal_correction
                right_corrected_gaze_vector[0] += horizontal_correction

                left_corrected_gaze_vector = (
                    left_corrected_gaze_vector[0] * horizontal_scale_factor ,
                    left_corrected_gaze_vector[1] * vertical_scale_factor ,
                    left_corrected_gaze_vector[2]
                )

                right_corrected_gaze_vector = (
                    right_corrected_gaze_vector[0] * horizontal_scale_factor ,
                    right_corrected_gaze_vector[1] * vertical_scale_factor ,
                    right_corrected_gaze_vector[2]
                )

                left_smoothed_corrected_gaze_vector = corrected_gaze_filter.update(left_corrected_gaze_vector)
                right_smoothed_corrected_gaze_vector = corrected_gaze_filter.update(right_corrected_gaze_vector)

                draw_corrected_gaze_vector(image, left_eye_center, left_smoothed_corrected_gaze_vector, vector_scale)
                draw_corrected_gaze_vector(image, right_eye_center, right_smoothed_corrected_gaze_vector, vector_scale)


                average_gaze_vector = (
                    (left_smoothed_corrected_gaze_vector[0] + right_smoothed_corrected_gaze_vector[0]) / 2,
                    (left_smoothed_corrected_gaze_vector[1] + right_smoothed_corrected_gaze_vector[1]) / 2,
                    (left_smoothed_corrected_gaze_vector[2] + right_smoothed_corrected_gaze_vector[2]) / 2
                )

                '''draw_corrected_gaze_vector(image, left_eye_center, average_gaze_vector,vector_scale)
                draw_corrected_gaze_vector(image, right_eye_center, average_gaze_vector,vector_scale)'''

                temp_gaze = average_gaze_vector


                if temp_gaze != (0, 0, 0):
                    convergence_point = (
                        int(image_center[0] + temp_gaze[0] * 100),
                        int(image_center[1] + temp_gaze[1] * 100)
                    )
                    cv2.drawMarker(image, convergence_point, (255, 0, 255), cv2.MARKER_CROSS)
                    text_position = (convergence_point[0], convergence_point[1] + 20)
                    cv2.putText(image, 'FOCUS POINT', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)



            else :
                if isinstance(temp_gaze, tuple) and len(temp_gaze) == 3 and all(
                        isinstance(num, (float, int)) for num in temp_gaze):
                    draw_corrected_gaze_vector(image, left_eye_center, temp_gaze)
                    draw_corrected_gaze_vector(image, right_eye_center, temp_gaze)
                    convergence_point = (
                        int(image_center[0] + temp_gaze[0] * 100),
                        int(image_center[1] + temp_gaze[1] * 100)
                    )
                    cv2.drawMarker(image, convergence_point, (255, 0, 255), cv2.MARKER_CROSS)
                    text_position = (convergence_point[0], convergence_point[1] + 20)
                    cv2.putText(image, 'FOCUS POINT', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    print("temp_gaze is not in the correct format:", temp_gaze)

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([quadratic_adjustment(temp_gaze[0]), quadratic_adjustment(temp_gaze[1])])

            # DRAW IRIS
            haut = (face_landmarks.landmark[left_pupil_pairs[0][0]].x * image.shape[1],
                    face_landmarks.landmark[left_pupil_pairs[0][0]].y * image.shape[0])
            bas = (face_landmarks.landmark[left_pupil_pairs[0][1]].x * image.shape[1],
                   face_landmarks.landmark[left_pupil_pairs[0][1]].y * image.shape[0])
            gauche = (face_landmarks.landmark[left_pupil_pairs[1][0]].x * image.shape[1],
                      face_landmarks.landmark[left_pupil_pairs[1][0]].y * image.shape[0])
            droite = (face_landmarks.landmark[left_pupil_pairs[1][1]].x * image.shape[1],
                      face_landmarks.landmark[left_pupil_pairs[1][1]].y * image.shape[0])

            diametre_vertical = np.linalg.norm(np.array(haut) - np.array(bas))
            diametre_horizontal = np.linalg.norm(np.array(gauche) - np.array(droite))
            rayon = int((diametre_vertical + diametre_horizontal) / 4)
            cv2.circle(image, (int(left_pupil_center[0]), int(left_pupil_center[1])), rayon, (0, 255, 255),1)

            haut = (face_landmarks.landmark[right_pupil_pairs[0][0]].x * image.shape[1],
                    face_landmarks.landmark[right_pupil_pairs[0][0]].y * image.shape[0])
            bas = (face_landmarks.landmark[right_pupil_pairs[0][1]].x * image.shape[1],
                   face_landmarks.landmark[right_pupil_pairs[0][1]].y * image.shape[0])
            gauche = (face_landmarks.landmark[right_pupil_pairs[1][0]].x * image.shape[1],
                      face_landmarks.landmark[right_pupil_pairs[1][0]].y * image.shape[0])
            droite = (face_landmarks.landmark[right_pupil_pairs[1][1]].x * image.shape[1],
                      face_landmarks.landmark[right_pupil_pairs[1][1]].y * image.shape[0])

            diametre_vertical = np.linalg.norm(np.array(haut) - np.array(bas))
            diametre_horizontal = np.linalg.norm(np.array(gauche) - np.array(droite))
            rayon = int((diametre_vertical + diametre_horizontal) / 4)
            cv2.circle(image, (int(right_pupil_center[0]), int(right_pupil_center[1])), rayon, (0, 255, 255), 1)

            # eyes center
            pairs_to_draw = [(33, 133), (159, 145), (463, 263), (386, 374)]
            draw_lines_between_points(image, face_landmarks, [(33, 133), (159, 145), (463, 263), (386, 374)],color=(255, 0, 0), thickness=1)

            # Head vector
            landmark_10 = (int(face_landmarks.landmark[10].x * image.shape[1]), int(face_landmarks.landmark[10].y * image.shape[0]))
            landmark_175 = (int(face_landmarks.landmark[175].x * image.shape[1]), int(face_landmarks.landmark[175].y * image.shape[0]))
            landmark_50 = (int(face_landmarks.landmark[50].x * image.shape[1]), int(face_landmarks.landmark[50].y * image.shape[0]))
            landmark_280 = (int(face_landmarks.landmark[280].x * image.shape[1]), int(face_landmarks.landmark[280].y * image.shape[0]))

            rotation_vector_3 = (landmark_10[0] - landmark_175[0], landmark_10[1] - landmark_175[1])
            rotation_vector_4 = (-rotation_vector_3[0], -rotation_vector_3[1])
            rotation_vector_1 = (landmark_50[0] - landmark_280[0], landmark_50[1] - landmark_280[1])
            rotation_vector_2 = (-rotation_vector_1[0], -rotation_vector_1[1])

            cv2.arrowedLine(image, landmark_50,(landmark_50[0] + rotation_vector_1[0], landmark_50[1] + rotation_vector_1[1]), (0, 255, 0),2)
            cv2.arrowedLine(image, landmark_280,(landmark_280[0] + rotation_vector_2[0], landmark_280[1] + rotation_vector_2[1]),(0, 255, 0), 2)
            cv2.arrowedLine(image, landmark_10,(landmark_10[0] + rotation_vector_3[0], landmark_10[1] + rotation_vector_3[1]), (0, 255, 0),2)
            cv2.arrowedLine(image, landmark_175,(landmark_175[0] + rotation_vector_4[0], landmark_175[1] + rotation_vector_4[1]),(0, 255, 0), 2)

    # Show the resulting image.
    cv2.imshow('MediaPipe FaceMesh', image)

    # Press 'q' to exit the loop.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources and close opened windows.
cap.release()
cv2.destroyAllWindows()
