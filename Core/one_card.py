import cv2
import numpy as np
import tensorflow as tf
from scipy import ndimage
from functions import *
from tflite_runtime.interpreter import load_delegate

size = (250,250)

cap = cv2.VideoCapture(0)

interpreter = tf.lite.Interpreter(model_path="../CNN_V2_git.tflite",experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

while True:
    _, frame = cap.read()
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    cropped,box = find_card(frame)
    cropped = cv2.resize(cropped,(160,160))
    cropped_cpy = cropped.copy()
    
    cropped = cropped.astype(np.float32)
    input_data = np.expand_dims(cropped, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    pred_id=np.argmax(prediction,axis=-1)
    pred_label=CLASS_NAMES[int(pred_id)]
    
    frame = cv2.resize(frame,size)
    x,y,w,h = box[0]
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    cv2.putText(frame,pred_label,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    frame = cv2.resize(frame,(750,750))
    cv2.imshow('frame',frame)
    #cv2.imshow('cropped',cropped)
    cv2.imshow('ROI',cropped_cpy)
    key = cv2.waitKey(1)
  
#closing all open windows 
cv2.destroyAllWindows() 




