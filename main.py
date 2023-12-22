import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

def add_new_face(image_path, name):
    new_face_image = face_recognition.load_image_file(image_path)
    new_face_encoding = face_recognition.face_encodings(new_face_image)[0]
    known_face_encodings.append(new_face_encoding)
    known_face_names.append(name)

faces_folder = "Faces"  # Replace with your folder path
for filename in os.listdir(faces_folder):
    name = os.path.splitext(filename)[0] 
    image_path = os.path.join(faces_folder, filename)
    add_new_face(image_path, name)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
frame_count = 0

while True:
    ret, frame = video_capture.read()
    frame_count += 1
    if frame_count % 3 == 0:  
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            if len(known_face_encodings) > 0:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    name = "Unknown"
            else:
                name = "No faces found"

            face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
