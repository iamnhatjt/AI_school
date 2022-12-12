# Python code to read image
import cv2

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

id=0
#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)
 
img = cv2.imread("binhthuong.jpg")
print(img.shape)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh camera
faces=faceDetect.detectMultiScale(gray,1.3,5)

for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(21,50,82),2)



 
cv2.imshow("image", img)
cv2.imwrite('dat.jpg', img)
 
# To hold the window user close it.
cv2.waitKey(0)
 
cv2.destroyAllWindows()