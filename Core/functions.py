import cv2
import numpy as np
from scipy import ndimage

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



def find_card(image):
  """performs card segmentation of one card from a given image
    @params : image : image matrix
    
    @returns: segmented card image"""

  #params
  index=0
  thickness=3
  color=(255,0,0) #color of the drawn contour
  size = (250,250)
  my_box = []

  
  image = cv2.resize(image,size) # resizing 

  image = cv2.GaussianBlur(image,(5,5),1) # gaussian blur to smoothen the image
  edge = cv2.Canny(image,100,200) # detect edges from colored image
  contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours
  
  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:]     
  
  #finding rectangle coordinates
  x,y,w,h = cv2.boundingRect(contours[0])
  my_box.append((x,y,w,h))
  #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
  #cv2_imshow(image)
  #rotated rectangle to crop the card and adjust it

  rect = cv2.minAreaRect(contours[0]) #create a rectangle from contour

  box = cv2.boxPoints(rect) #convert into a four points array[[x y]]
  box = np.int0(box) #cast into integers

  angle = rect[2]

  mask = np.zeros(image.shape, np.uint8)
  cv2.drawContours(mask, [box], -1, (255, 255, 255), -1, cv2.LINE_AA)
  result = cv2.bitwise_and(image, mask)
  rotated = ndimage.rotate(result, angle)
  trimmedImage = trim(rotated)
  trimmedImage = cv2.resize(trimmedImage,(250,250))

  return trimmedImage,my_box



def find_cards(image):
  """performs segmentation of all cards present on the image
     @params : image_path
     @returns: an array containing the segmented images"""
  #params
  index=-1
  thickness=3
  color=(255,0,0)
  seuil = 0.5
  cards = []
  box = []

  #image = cv2.imread(image_path)
  image = cv2.resize(image,(750,750)) # resizing 
  #creating image copies that we will be drawn on
  
  contours_image = image.copy() #the copy that will contain the contours (bleu)
  box_image = image.copy()      #the copy that will contain the boxes 

  #finding contours
  image = cv2.GaussianBlur(image,(5,5),1) # gaussian blur to smoothen the image
  edge = cv2.Canny(image,100,200) # detect edges from colored image
  contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find contours

  #sorting contours by surface
  contours = sorted(contours, key=cv2.contourArea,reverse=True)[:]

  #selecting a threshold and filtering noise
  max = cv2.contourArea(contours[0])
  thresh = max*seuil
  contours = [c for c in contours if cv2.contourArea(c)>thresh]

  #drawing contours on on the contours_image
  #cv2.drawContours(contours_image,contours,index,color,thickness)
 
  #from contours to bounding boxes

  rect = [cv2.minAreaRect(i) for i in contours] #getting rectangle
  

  for i in range(0,len(contours)):
    _ctr = contours[i]
    x,y,w,h = cv2.boundingRect(_ctr)

    box.append((x,y,w,h))
    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

  #rect = cv2.minAreaRect(_ctr)
  #box = cv2.boxPoints(rect)
  #box = np.int0(box)
  #cv2.drawContours(image,[box],0,(0,0,255),2)

  

    cropped = image[y:y+h,x:x+w]
    cards.append(cropped)
    
  return cards,box
  
