import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import time
from pymongo import MongoClient   # <--- NEW

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["driver_monitor_db"]
collection = db["results"]

# Initialize the mixer and alert sound
mixer.init()
mixer.music.load("static/alert.wav")

# EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR (Mouth Aspect Ratio)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)

# Thresholds
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 10
MAR_THRESH = 0.7
ALERT_DURATION = 2  # seconds
flag = 0
last_alert_time = 0

# Facial landmarks
FACIAL_LANDMARK_68_IDXS = {
    "mouth": (48, 68),
    "right_eye": (36, 42),
    "left_eye": (42, 48)
}

(lStart, lEnd) = FACIAL_LANDMARK_68_IDXS['left_eye']
(rStart, rEnd) = FACIAL_LANDMARK_68_IDXS['right_eye']
(mStart, mEnd) = FACIAL_LANDMARK_68_IDXS['mouth']

# Load detectors
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"D:\driver_monitor_web\shape_predictor_68_face_landmarks.dat")

def generate_frames():
    global flag, last_alert_time
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        status_text = "Normal"

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            # Draw contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 0, 255), 1)

            # Drowsiness detection
            if ear < EAR_THRESH:
                flag += 1
                if flag >= EAR_CONSEC_FRAMES:
                    status_text = "Drowsy!"
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if time.time() - last_alert_time > ALERT_DURATION:
                        mixer.music.play()
                        last_alert_time = time.time()
                        # Store event in MongoDB
                        collection.insert_one({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Drowsy"
                        })
            else:
                flag = 0

            # Yawning detection
            if mar > MAR_THRESH:
                status_text = "Yawning!"
                cv2.putText(frame, "YAWNING ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if time.time() - last_alert_time > ALERT_DURATION:
                    mixer.music.play()
                    last_alert_time = time.time()
                    # Store event in MongoDB
                    collection.insert_one({
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "Yawning"
                    })

        cv2.putText(frame, f"Status: {status_text}", (10, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
