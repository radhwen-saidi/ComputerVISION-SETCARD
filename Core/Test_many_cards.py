import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
from functions import *
from tflite_runtime.interpreter import load_delegate
import time
size = (500,500)
pred_label = []


interpreter = tf.lite.Interpreter(model_path="../CNN_V2_git.tflite",experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


#test zone ################

path = "/home/pi/Desktop/Modèles/script/img.png"


#end of test zone ######################

while True:
    
    frame = cv2.imread(path)
    
    frame = cv2.resize(frame,size)
   
    
    cards,box = find_cards(frame)
    
    cards = [cv2.resize(card,(160,160)) for card in cards]
    cards = [card.astype(np.float32) for card in cards]
    cards = [cv2.cvtColor(card,cv2.COLOR_BGR2RGB) for card in cards]
    input_data_vector = [np.expand_dims(card, axis=0) for card in cards]
    
    for input_data in input_data_vector:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        pred_id=np.argmax(prediction,axis=-1)
        pred_label.append(CLASS_NAMES[int(pred_id)])
        
    frame = cv2.resize(frame,(750,750))
    for b in box:
        x,y,w,h = b
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(frame,pred_label[box.index(b)],(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('frame ',frame)
    #cv2.imshow('cropped',cropped)
    #cv2.imshow('ROI',cropped_cpy)
    key = cv2.waitKey(1)
    
    
#closing all open windows 
cv2.destroyAllWindows() 




