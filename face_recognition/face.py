
# https://github.com/ageitgey/face_recognition

# gswewf@gswewf-pc:~$ sudo apt-get install libboost-dev
# gswewf@gswewf-pc:~$ sudo apt-get install libboost-python-dev 
# gswewf@gswewf-pc:~$ sudo pip3 install face_recognition

import face_recognition
image = face_recognition.load_image_file("your_file.jpg")
face_locations = face_recognition.face_locations(image)


