import numpy as np
import cv2
import time
import face_alignment

from feature_detection import *

# Visualization of landmark
# ear for blink detection

# Initialize the face alignment tracker
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device="cuda")

# Start the webcam capture, exit with 'q'
cap = cv2.VideoCapture("chemistry2.mp4")
while (not (cv2.waitKey(1) & 0xFF == ord('q'))):
    ret, frame = cap.read()
    if (ret):
        # Clear the indices frame
        canonical = np.zeros(frame.shape)

        # Run the face alignment tracker on the webcam image
        imagePoints = fa.get_landmarks_from_image(frame)
        if (imagePoints is not None):
            imagePoints = imagePoints[0]

            # Compute the Mean-Centered-Scaled Points
            mean = np.mean(imagePoints, axis=0)  # <- This is the unscaled mean
            # scaled = (imagePoints / np.linalg.norm(
            #     imagePoints[42] - imagePoints[39])) * 0.06  # Set the inner eye distance to 60cm (just because)
            # centered = scaled - np.mean(scaled, axis=0)  # <- This is the scaled mean
            centered = (imagePoints - mean) * 0.002

            # Construct a "rotation" matrix (strong simplification, might have shearing)
            rotationMatrix = np.empty((3, 3))
            rotationMatrix[0, :] = (centered[16] - centered[0]) / np.linalg.norm(centered[16] - centered[0])
            rotationMatrix[1, :] = (centered[8] - centered[27]) / np.linalg.norm(centered[8] - centered[27])
            rotationMatrix[2, :] = np.cross(rotationMatrix[0, :], rotationMatrix[1, :])
            invRot = np.linalg.inv(rotationMatrix)

            # Object-space points, these are what you'd run OpenCV's solvePnP() with
            objectPoints = centered.dot(invRot)
            # objectPoints = (objectPoints / np.linalg.norm(
            #     objectPoints[42] - objectPoints[39])) * 0.06

            left_ear = eye_aspect_ratio(objectPoints[36:42])
            right_ear = eye_aspect_ratio(objectPoints[42:48])
            ear = (left_ear + right_ear) / 2
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    blinked = True
                COUNTER = 0

            objectPoints_2d = objectPoints[:, :-1]
            # Draw the computed data
            for i, (imagePoint, objectPoint) in enumerate(zip(imagePoints, objectPoints)):
                # Draw the Point Predictions
                cv2.circle(frame, (imagePoint[0], imagePoint[1]), 3, (0, 255, 0))

                # Draw the X Axis
                cv2.line(frame, tuple(mean[:2].astype(int)),
                         tuple((mean + (rotationMatrix[0, :] * 100.0))[:2].astype(int)), (0, 0, 255), 3)
                # Draw the Y Axis
                cv2.line(frame, tuple(mean[:2].astype(int)),
                         tuple((mean - (rotationMatrix[1, :] * 100.0))[:2].astype(int)), (0, 255, 0), 3)
                # Draw the Z Axis
                cv2.line(frame, tuple(mean[:2].astype(int)),
                         tuple((mean + (rotationMatrix[2, :] * 100.0))[:2].astype(int)), (255, 0, 0), 3)

                # Draw the indices in Object Space
                cv2.putText(canonical, str(i),
                            ((int)((objectPoint[0] * 1000.0) + 320.0),
                             (int)((objectPoint[1] * 1000.0) + 240.0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Webcam View', frame)
        cv2.imshow('Canonical View', canonical)

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()