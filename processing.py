from typing import Union, Any, Optional, Tuple, List

import mediapipe as mp
import cv2
import math

from cv2 import Mat
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python import solutions
from numpy import ndarray, dtype, generic

# Set up MediaPipe solutions configurations.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

custom_hand_landmark_drawing_style = mp_drawing_styles.get_default_hand_landmarks_style()

# List of indices for each angle we want.
DEFAULT_ANGLE_INDICES = [
    (0, 1, 2),  # 1
    (1, 2, 3),  # 2
    (2, 3, 4),  # 3
    (0, 5, 6),  # 4
    (5, 6, 7),  # 5
    (6, 7, 8),  # 6
    (0, 5, 9),  # 7
    (5, 9, 10),  # 8
    (9, 10, 11),  # 9
    (10, 11, 12),  # 10
    (5, 9, 13),  # 11
    (9, 13, 14),  # 12
    (13, 14, 15),  # 13
    (14, 15, 16),  # 14
    (13, 17, 18),  # 15
    (17, 18, 19),  # 16
    (18, 19, 20),  # 17
    (0, 17, 18)  # 18
]


def calculate_angles(points: list[tuple[float, float, float]], angle_indices: list[tuple[int, int, int]] = None) -> \
        list[float]:
    """
    Calculates a list of angles based on a list of points.
    :param points: A list of 21 tuples, containing the coordinates of each landmark.
    :param angle_indices: The list of indices in points to use.
    :return: The list of angles calculated from the points.
    """

    if angle_indices is None:
        angle_indices = DEFAULT_ANGLE_INDICES

    angles = []

    # Iterate over each tuple of indices, and calculate the angle between the resulting vectors.
    for (i, j, k) in angle_indices:
        point1, point2, point3 = points[i], points[j], points[k]

        # Create the vectors from the points.
        vector1 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2])

        # Find the dot product of the two vectors.
        dot_product = (vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2])

        # Calculate the magnitudes of the vectors.
        vector1_magnitude = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2 + vector1[2] ** 2)
        vector2_magnitude = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2 + vector2[2] ** 2)

        # From the dot product and magnitudes, find the angle in radians.
        angle = math.acos(dot_product / (vector1_magnitude * vector2_magnitude))

        angles.append(angle)

    return angles


def process_frame(frame: Union[cv2.Mat, ndarray[Any, dtype[generic]], ndarray], confidence: float = 0.5):
    """
    Processes a frame from the webcam, captured from cv2. Assume that the amount of detected hands will always be 1, if
    hand(s) are detected. Assert that something alike "hand_landmarks" refers to "multi_hand_landmarks[0]".
    :param frame: The frame to predict process.
    :param confidence: The confidence (handedness) to process the frame with.
    :return: The angles of the fingers.
    """

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=confidence,
            model_complexity=0  # Subject to Change.
    ) as hands:

        # Flip the image, and change the colors from BGR to RGB.
        image = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Check if the frame could not find a hand with a high enough confidence.
        if not results.multi_hand_landmarks:
            return None, image

        # Isolate the coordinates of the landmarks.
        landmarks = [(pos.x, pos.y, pos.z) for pos in results.multi_hand_world_landmarks[0].landmark]

        return calculate_angles(landmarks), annotate_image(image, results)


def annotate_image(image, results):

    """
    Annotates an image by drawing the hand landmarks and their respective connections.
    :param image: The image to annotate.
    :param results: The set of hand landmark data.
    :return: The annotated image.
    """

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.multi_hand_landmarks[0],
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )
    return annotated_image
