import cv2
#load some pre-trained data on face frontals from opencv (harr cascade alogrithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#chose an image to detect face need to change for every file
img = cv2.imread('Niklas.jpeg')


#converting to gray scale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect face
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#print(face_coordinates)
#draw rectangles
(x,y,w,z) = face_coordinates[0]
cv2.rectangle(img,(x, y), (x+w, y+z), (0, 255, 0), 2)
#
cv2.imshow('Clever programmer face detector', img)
cv2.waitKey()


print("code completed")