import numpy as np #np for shortcut 
import face_recognition as fr #fr for shortcut
import cv2 

video_capture = cv2.VideoCapture(0) #capturing the video from webcam 1-off 0-on

taran_image = fr.load_image_file("taran.jpeg") #load the picture 
taran_face_encoding = fr.face_encodings(taran_image)[0] #analyse the picture and do encoding of the face and will return array for number of faces

known_face_encondings = [taran_face_encoding] #array for known faces
known_face_names = ["taran"] #name displayed when face is detected

while True: #created the loop to take all the frameworks of camera
    ret, frame = video_capture.read() #ret is boolean which tells whether there was a return or not
                                      #frame will get the next frame via capture

    rgb_frame = frame[:, :, ::-1] #color of the frames, open CV uses BGR format for coloring but this command changes to RGB
                                
    face_locations = fr.face_locations(rgb_frame) #where are the faces in this frame
    face_encodings = fr.face_encodings(rgb_frame, face_locations) #then faces are encoded

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings): #iterate over all the locations and encodings
                                                                                          #zip helps to iterate between two of them
        matches = fr.compare_faces(known_face_encondings, face_encoding) #matches the scanned faces with known faces

        name = "Unknown" #name of the unknown face detected

        face_distances = fr.face_distance(known_face_encondings, face_encoding) # compare the faces and check the euclidean distance

        best_match_index = np.argmin(face_distances) #if distance is more face doesn't resemble much but if it is less the face resembles more
        if matches[best_match_index]: #if condition for matching the face
            name = known_face_names[best_match_index] # Then show the name of the matching face
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)#customizing the frame and BGR color which is going to be formed on the face 

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED) #wiil make a small red fill box over which name will be written
        font = cv2.FONT_HERSHEY_SIMPLEX #font for the text 
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 3)# text which is written below the frame

    cv2.imshow('Webcam_facerecognition', frame) #we need to show the image

    if cv2.waitKey(1) & 0xFF == ord('q'): #if want to escape from the face recognoition window just press key q
        break

video_capture.release() #it will stop taking the video from the webcam 
cv2.destroyAllWindows() #destroy everything(in tab) after we quit q