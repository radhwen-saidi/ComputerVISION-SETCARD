import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
from functions import *
from tflite_runtime.interpreter import load_delegate
import os 

#variables
size = (750,750) #Size of images that will be 
card_size = (160,160)
pred_label = []  #Table containing the predicted labels
model_path="../CNN_V2_git.tflite" #path to the trained model (tensorflow lite)

#opening camera 
cap = cv2.VideoCapture(0) 


interpreter = tf.lite.Interpreter(model_path,experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details() 
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]


#setting height and width parameters of the camera input

input_height = 1024 
input_width  = 1280 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(input_width))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(input_height))

#endless loop
while True:
    
    #resetting variables to avoid 
    pred_label = []
    cards = []
    
    #reading image from camera
    ret, frame = cap.read()
    frame = cv2.resize(frame,size)
    
    #segmenting cards from the image
    try:
        cards,box = find_cards(frame)
    except IndexError:
        print("cards not found! please put cards in the deck!")
        continue
    
    
    #converting cards to a compatible format with the model input (1,160,160,3)
    
    cards = [cv2.resize(card,card_size) for card in cards]
    cards = [card.astype(np.float32) for card in cards]
    cards = [cv2.cvtColor(card,cv2.COLOR_BGR2RGB) for card in cards]
    input_data_vector = [np.expand_dims(card, axis=0) for card in cards]
    
    #for each card, get the prediction (can be enhanced)
    
    for input_data in input_data_vector:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        pred_id=np.argmax(prediction,axis=-1)
        pred_label.append(CLASS_NAMES[int(pred_id)])
    
    #resizing frame to frame size
    frame = cv2.resize(frame,size)
    
    #drawing boxes & label on each card
    for b in box:
        x,y,w,h = b
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv2.putText(frame,pred_label[box.index(b)],(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    #showing result
    cv2.imshow('frame',frame)
    
    key = cv2.waitKey(1)
    
#closing all open windows 
cv2.destroyAllWindows() 




