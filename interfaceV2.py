import tkinter
import cv2
import PIL.Image, PIL.ImageTk
from PIL import Image
import time
import numpy as np
import base64
from tflite_runtime.interpreter import load_delegate
import tensorflow as tf


class App:
     def __init__(self, window, window_title, video_source=0):
         self.window = window
         self.window.title(window_title)
         self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
         self.vid = MyVideoCapture(self.video_source)
         '''pre_proc = Cards.preprocess_image(self.vid)'''

        # Create a canvas that can fit the above video source size
         self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
         self.canvas.grid(row=1, column=0)
         self.frame1 = tkinter.Frame(window, width = self.vid.width/2, height = self.vid.height/2)
         self.frame1.grid(row=1, column=1)
 
        # Creation des elements de la fenetre
         global labelTexte
         labelTexte=tkinter.Label(self.frame1, text="Precision", width=50)
         labelTexte.grid(row=1, column=0)

         self.labelPrec=tkinter.Label(self.frame1, text="95%", width=50)
         self.labelPrec.grid(row=2, column=0)
         
         chemin ="/home/pi/Desktop/Interface/1-green-empty-diamond.png"
         imageInitiale = tkinter.PhotoImage(file=chemin).zoom(32).subsample(64)
         '''global canvas2
         canvas2 = tkinter.Button(self.frame1,image=imageInitiale, width = self.vid.width/2, height = self.vid.height/2, bg="white")
         #imageCard = Image.open("D:/S7/Edge_Computing/Interface/cardsE/roi.png")
         canvas2.grid(row=0, column=0)'''

         global labelImage
         labelImage=tkinter.Label(self.frame1, image=imageInitiale, width=500, height=500)
         labelImage.grid(row=0, column=0)

         # 

         def rafraichir():
             ret, frame, name = self.vid.get_frame()
             cardName, prec = name.split(':')
             labelPrec.config(text=prec)
             #cardName = "roi.png"
             chemin = "/home/pi/Desktop/Interface/cartes/"+cardName+".png"
             imageCarte = tkinter.PhotoImage(file=chemin)
             labelImage.configure(image=imageCarte)
             labelImage.after(100, retour)
             labelImage.image=imageCarte
             
        #Appel de la fct rafraichir pour rafraichir les données
         rafraichir()

        # After it is called once, the update method will be automatically called every delay milliseconds
         self.delay = 150
         self.update()
 
         self.window.mainloop()

 
    
     def update(self):
        # Get a frame from the video source
        ret, frame, name = self.vid.get_frame()
 
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0,0 , image = self.photo, anchor = tkinter.NW)
 
        self.window.after(self.delay, self.update)
 

 
class MyVideoCapture:
     def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
 
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

     def trim(frame):
        #crop top
        if not np.sum(frame[0]):
            return trim(frame[1:])
        #crop bottom
        elif not np.sum(frame[-1]):
            return trim(frame[:-2])
        #crop left
        elif not np.sum(frame[:,0]):
            return trim(frame[:,1:]) 
        #crop right
        elif not np.sum(frame[:,-1]):
            return trim(frame[:,:-2])    
        return frame

     def find_ROI(image):
        """Without plotting the steps, finds the contours and crops the image"""

        #image = cv2.imread(image_path)
        image = cv2.resize(image,(250,250)) # resizing 
        #cv2_imshow(image)
        image = cv2.GaussianBlur(image,(5,5),1) # gaussian blur to smoothen the image
        edge = cv2.Canny(image,100,200) # detect edges from colored image

        contours, hierarchy = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours

        img_cpy = image.copy() #create an image copy
        
        index=0
        thickness=3
        color=(255,0,0)
        
        cv2.drawContours(img_cpy,contours,index,color,thickness)  #select first contour
        #cv2_imshow(edge) #show edged image
        #cv2_imshow(img_cpy) #show image + contours

        ############################### min surf rect
        img_cpy3 = image.copy()
        rect = cv2.minAreaRect(contours[0]) #create a rectangle from contour

        box = cv2.boxPoints(rect) #convert into a four points array[[x y]]
        box = np.int0(box) #cast into integers

        angle = rect[2]
        #print("theta= ",angle)
        im = cv2.drawContours(img_cpy3,[box],0,(0,0,255),2) #draw rectangular contour
        #cv2_imshow(im)

        rotated = ndimage.rotate(img_cpy3, angle)
        #cv2_imshow(rotated)

        pts = box

        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        result = cv2.bitwise_and(image, mask)
        rotated = ndimage.rotate(result, angle)

        #cv2_imshow(rotated)

        
        thold = (rotated>120)*rotated
        trimmedImage = trim(rotated)
        trimmedImage = cv2.resize(trimmedImage,(160,160))
        
        #cv2_imshow(thold)
        #cv2_imshow(trimmedImage)
        return trimmedImage
 
     def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                CLASS_NAMES = ['2-purple-empty-squiggles',
                '2-red-striped-ovals',
                '2-green-empty-squiggles',
                '1-green-striped-squiggle',
                '3-red-empty-diamonds',
                '3-red-empty-squiggles',
                '2-green-solid-squiggles',
                '3-green-empty-squiggles',
                '1-purple-striped-oval',
                '2-green-striped-squiggles',
                '2-purple-solid-squiggles',
                '1-purple-empty-oval',
                '2-purple-empty-ovals',
                '2-purple-striped-ovals',
                '3-red-striped-diamonds',
                '3-purple-striped-diamonds',
                '3-green-solid-diamonds',
                '2-red-striped-squiggles',
                '3-green-solid-squiggles',
                '2-red-solid-diamonds',
                '1-purple-empty-diamond',
                '1-purple-empty-squiggle',
                '2-red-solid-ovals',
                '3-purple-solid-diamonds',
                '2-red-empty-diamonds',
                '2-purple-solid-ovals',
                '1-red-striped-diamond',
                '1-red-striped-oval',
                '3-green-solid-ovals',
                '3-purple-empty-squiggles',
                '2-green-empty-diamonds',
                '3-red-solid-squiggles',
                '1-red-solid-oval',
                '2-red-empty-ovals',
                '1-purple-solid-oval',
                '2-purple-striped-diamonds',
                '1-green-empty-oval',
                '1-green-solid-oval',
                '2-purple-empty-diamonds',
                '3-purple-striped-ovals',
                '3-purple-empty-diamonds',
                '2-green-solid-diamonds',
                '3-green-striped-ovals',
                '1-red-empty-oval',
                '2-purple-solid-diamonds',
                '2-green-striped-ovals',
                '1-green-solid-diamond',
                '2-purple-striped-squiggles',
                '3-purple-solid-ovals',
                '2-green-empty-ovals',
                '2-green-solid-ovals',
                '3-red-solid-diamonds',
                '1-green-striped-diamond',
                '1-green-empty-squiggle',
                '1-red-solid-diamond',
                '3-red-empty-ovals',
                '3-green-striped-squiggles',
                '1-purple-striped-diamond',
                '2-red-empty-squiggles',
                '1-green-empty-diamond',
                '3-green-empty-ovals',
                '1-red-empty-squiggle',
                '1-purple-striped-squiggle',
                '1-green-striped-oval',
                '3-purple-solid-squiggles',
                '1-red-empty-diamond',
                '3-green-empty-diamonds',
                '1-red-solid-squiggle',
                '3-red-striped-squiggles',
                '2-red-striped-diamonds',
                '3-green-striped-diamonds',
                '3-purple-striped-squiggles',
                '2-red-solid-squiggles',
                '3-red-striped-ovals',
                '1-purple-solid-diamond',
                '1-green-solid-squiggle',
                '2-green-striped-diamonds',
                '1-red-striped-squiggle',
                '3-purple-empty-ovals',
                '1-purple-solid-squiggle',
                '3-red-solid-ovals']

                

                interpreter = tf.lite.Interpreter(model_path="/home/pi/Desktop/Interface/CNN_V2_git.tflite",experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                height = input_details[0]['shape'][1]
                width = input_details[0]['shape'][2]
                cropped = find_ROI(frame)
                ROI = cropped.copy()
                cropped = cropped.astype(np.float32)
                input_data = np.expand_dims(cropped, axis=0)
                    
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                pred_id=np.argmax(prediction,axis=-1)
                pred_label=CLASS_NAMES[int(pred_id)]
                PC=prediction[0][pred_id]*100
                name= str(pred_label) + ":" + str(PC)
                print(pred_label)
                    
                cv2.imshow('frame',frame)
                #cv2.imshow('cropped',cropped)
                cv2.imshow('ROI',ROI)        
                cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),0)
          	 
                return (ret, ROI,name)
                #OU return (ret, frame,name)
            else:
                return (ret, None)
        else:
            return (ret, None)
 
    # Release the video source when the object is destroyed
     def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
 # Create a window and pass it to the Application object
window = tkinter.Tk()
App(window, "Tkinter and OpenCV")
