from typing import Union, Any

import mediapipe as mp
import cv2
import math

from numpy import ndarray, dtype, generic

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

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

    for (i, j, k) in angle_indices:
        point1, point2, point3 = points[i], points[j], points[k]

        vector1 = (point2[0] - point1[0], point2[1] - point1[1], point2[2] - point1[2])
        vector2 = (point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2])

        dot_product = (vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2])

        vector1_magnitude = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2 + vector1[2] ** 2)
        vector2_magnitude = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2 + vector2[2] ** 2)

        angle = math.acos(dot_product / (vector1_magnitude * vector2_magnitude))

        angles.append(angle)

    return angles


def process_frame(frame: Union[cv2.Mat, ndarray[Any, dtype[generic]], ndarray], confidence: float = 0.5) -> list[float]:
    """
    Processes a frame from the webcam, captured from cv2.
    :param frame: The frame to predict process.
    :param confidence: The confidence (handedness) to process the frame with.
    :return: The angles of the fingers.
    """

    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=confidence
    ) as hands:
        image = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            return None

        if not results.multi_hand_world_landmarks:
            return None

        landmarks = [(pos.x, pos.y, pos.z) for pos in results.multi_hand_world_landmarks[0].landmark]

        return calculate_angles(landmarks)