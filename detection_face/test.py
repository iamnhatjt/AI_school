
# Python code to read image
import cv2

 
# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread("D:\\Learning\\school\\Ai\\code\\detection_face\\binhthuong.jpg", cv2.IMREAD_COLOR)

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh camera
faces=faceDetect.detectMultiScale(gray,1.3,5)

for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(21,50,82),16)



scale_percent = 30 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", resized)
 
# To hold the window on screen, we use cv2.waitKey method
# Once it detected the close input, it will release the control
# To the next line
# First Parameter is for holding screen for specified milliseconds
# It should be positive integer. If 0 pass an parameter, then it will
# hold the screen until user close it.
cv2.imwrite('son.jpg', img)
cv2.waitKey(0)
 
# It is for removing/deleting created GUI window from screen
# and memory
cv2.destroyAllWindows()