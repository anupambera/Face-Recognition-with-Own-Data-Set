import face_recognition
import cv2
import numpy as np

import os
'''
    Get current working director and create a Data directory to store the faces
'''
currentDirectory = os.getcwd()
dirName = os.path.join(currentDirectory, 'Data')
print(dirName)
if not os.path.exists(dirName):
    try:
        os.makedirs(dirName)
    except:
        raise OSError("Can't create destination directory (%s)!" % (dirName))
'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def knownFaceEncoding(listOfFiles):
    known_face_encodings=list()
    known_face_names=list()
    for file_name in listOfFiles:
        # print(file_name)
        if(file_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
            known_image = face_recognition.load_image_file(file_name)
            # known_face_locations = face_recognition.face_locations(known_image)
            # known_face_encoding = face_recognition.face_encodings(known_image,known_face_locations)
            face_encods = face_recognition.face_encodings(known_image)
            if face_encods:
                known_face_encoding = face_encods[0]
                known_face_encodings.append(known_face_encoding)
                known_face_names.append(os.path.basename(file_name[0:-4]))
    return known_face_encodings, known_face_names


# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
known_face_encodings, known_face_names = knownFaceEncoding(listOfFiles)

video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Video", flags= cv2.WINDOW_NORMAL)
# cv2.namedWindow("Video")

cv2.resizeWindow('Video', 1024,640)
cv2.moveWindow('Video', 20,20)


# known_face_encodings=list()
# known_face_names=list()
# for file_name in listOfFiles:
#     print(file_name)
#     if(file_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
#         known_image = face_recognition.load_image_file(file_name)
#         # known_face_locations = face_recognition.face_locations(known_image)
#         # known_face_encoding = face_recognition.face_encodings(known_image,known_face_locations)
#         face_encods = face_recognition.face_encodings(known_image)
#         if face_encods:
#             known_face_encoding = face_encods[0]
#             known_face_encodings.append(known_face_encoding)
#             known_face_names.append(os.path.basename(file_name[0:-12]))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # print(ret)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]


    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k== ord('c'):
        face_loc = face_recognition.face_locations(rgb_small_frame)
        if face_loc:
            print("Enter Name -")
            name = input()
            img_name = "{}/{}.png".format(dirName,name)
            (top, right, bottom, left)= face_loc[0]
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.imwrite(img_name, frame[top - 5 :bottom + 5,left -5 :right + 5])
            listOfFiles = getListOfFiles(dirName)
            known_face_encodings, known_face_names = knownFaceEncoding(listOfFiles)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # print(face_locations)

        face_names = []

        for face_encoding,face_location in zip(face_encodings,face_locations):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance= 0.55)
            name = "Unknown"
            distance = 0

            # use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #print(face_distances)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # distance = face_distances[best_match_index]
            #print(face_distances[best_match_index])
            # string_value = '{} {:.3f}'.format(name, distance)
            face_names.append(name)


    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom + 46), (right, bottom+11), (0, 0, 155), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom +40), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
