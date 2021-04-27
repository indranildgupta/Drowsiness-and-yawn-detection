import cv2
import dlib
from scipy.spatial import distance
import winsound

def aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio


cap = cv2.VideoCapture('test2.mp4')

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frequency = 500
duration = 200

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = dlib_facelandmark(gray, face)

        left_eye=[] #37--42
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x,y))
            n1=n+1
            if(n==41):
            	n1=36
            x1 = landmarks.part(n1).x
            y1 = landmarks.part(n1).y
            cv2.line(frame, (x, y), (x1,y1), (255, 0, 0), 1)

        right_eye=[] #43--48
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x,y))
            n1=n+1
            if(n==47):
            	n1=42
            x1 = landmarks.part(n1).x
            y1 = landmarks.part(n1).y
            cv2.line(frame, (x, y), (x1,y1), (255, 0, 0), 1)

        left_ear = aspect_ratio(left_eye)
        right_ear = aspect_ratio(right_eye)

        EAR = (left_ear+right_ear)/2

        if EAR < 0.2:
            cv2.putText(frame,"DROWSINESS",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,128,255),4)
            winsound.Beep(frequency,duration)
            #print("Drowsy")
        #print(EAR)

    cv2.imshow("Drowsiness and Yawn detection system", frame)

    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

