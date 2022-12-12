import cv2
import numpy as np

# Khởi tạo bộ phát hiện khuôn mặt
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

# Khởi tạo bộ nhận diện khuôn mặt



id=0
#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)

# Hàm lấy thông tin người dùng qua ID

# Khởi tạo camera
cam=cv2.VideoCapture(0);

while(True):

    # Đọc ảnh từ camera
    ret,img=cam.read();

    # Lật ảnh cho đỡ bị ngược
    img = cv2.flip(img, 1)

   

    # Chuyển ảnh về xám
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt trong ảnh camera
    faces=faceDetect.detectMultiScale(gray,1.3,5);

    # Lặp qua các khuôn mặt nhận được để hiện thông tin
    for(x,y,w,h) in faces:
        # Vẽ hình chữ nhật quanh mặt
        cv2.rectangle(img,(x,y),(x+w,y+h),(21,50,82),2)

        # Nhận diện khuôn mặt, trả ra 2 tham số id: mã nhân viên và dist (dộ sai khác)




    cv2.imshow('Face',img)
    # Nếu nhấn q thì thoát
    if cv2.waitKey(1)==ord('q'):
        cv2.imwrite('nhat.jpg', img)
        print('oke')
        break
        
cam.release()
cv2.destroyAllWindows()

