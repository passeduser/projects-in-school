import cv2
from cv2 import COLOR_BGR2RGB
import numpy as np
import face_recognition as fc
import dlib

#Image to make learn
eln_img = fc.load_image_file("elon.jpg")#importing image using face_recog..
eln_img = cv2.cvtColor(eln_img, COLOR_BGR2RGB)#converting bgr to rgb

#image to test
eln_test = fc.load_image_file("elon_test.jpg")
eln_test = cv2.cvtColor(eln_test, COLOR_BGR2RGB)


face_location = fc.face_locations(eln_img)[0]
encoding_image = fc.face_encodings(eln_img)[0]
cv2.rectangle(eln_img, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (900, 44, 900), 2)

#to test
face_location_test = fc.face_locations(eln_test)[0]
encoding_image_test = fc.face_encodings(eln_test)[0]
cv2.rectangle(eln_test, (face_location_test[3], face_location_test[0]), (face_location_test[1], face_location_test[2]), (255, 0, 255), 2)
 
#last step comparing these faces
results = fc.compare_faces([encoding_image], encoding_image_test)
face_dis = fc.face_distance([encoding_image], encoding_image_test)
print(results, face_dis)
cv2.putText(eln_test, f"{results}, {round(face_dis[0], 2)}", (50,50), cv2.FONT_ITALIC,1,(10, 10, 200),2)

cv2.imshow("Eln img", eln_img)
cv2.imshow("Eln img test", eln_test) 

# cv2.imshow("img 2", eln_test)
cv2.waitKey(0)