import cv2 
from cv2 import COLOR_BGR2RGB 
import face_recognition as fc
import os

path = r"Image_Attendance/"
images = []
class_list = []
mylist = os.listdir(path)
print(mylist)

# eln_img = fc.load_image_file("elon.jpg")
# eln_img = cv2.cvtColor(eln_img, COLOR_BGR2RGB)
# eln_test = fc.load_image_file("test.jpg")
# eln_test = cv2.cvtColor(eln_test, COLOR_BGR2RGB)

for cls in mylist:
    cur_list = cv2.imread(f"{path}/{cls}")
    images.append(cur_list)
    class_list.append(os.path.splitext(cls)[0])
print(class_list)

def findEncodings(image):
    encodelist = []
    for img in image:
        img = cv2.cvtColor(img, COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodings = findEncodings(images)
print("encoding complete")


capture = cv2.VideoCapture(0)
while True:
    success, img = capture.read()
    imgS= cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, COLOR_BGR2RGB)
    face_loc_current = fc.face_locations(imgS)[0]
    encodings_cur_frame = fc.face_encodings(imgS, face_loc_current)

    for encodeFace, faceLoc in zip(encodings_cur_frame, face_loc_current):
        pass
    '''more next time'''