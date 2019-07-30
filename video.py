import cv2
import numpy as np
import time
import math
from plate_detection import plate
from character_detection import character
from classCNN import NeuralNetwork

cap = cv2.VideoCapture("test_videos/test2.MOV")
myNetwork = NeuralNetwork(modelFile="model/ducdn_ver3.pb",labelFile="model/ducdn_ver3.txt")
def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
    return num 

def getDistance(x, y, x1, y1):
    return math.sqrt(math.pow((x-x1), 2) + math.pow((y-y1), 2))

List_plate = []
len_plate = []
list_compare = []
list_string = []
X = []
Y = [] 
comparison_array = []
change_1D_array = []
change_2D_array = []
compare_plate = []
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if (frame is None):
        print("[INFO] End of Video")
        break
    #cv2.imshow('frame',frame)
    frame = frame[400:1080,0:1400]
    frame = cv2.resize(frame, (1600,1200))

    identifyPlates = plate(frame, type_of_plate = 'long_plate')
    morp = identifyPlates.process(frame)
    contour = identifyPlates.extract_contours(morp)
    if contour:
        try:
            clean_plate, x, y, w, h = identifyPlates.CleanAndRead(contour)
            identifyCharacter = character(clean_plate)
            thresh, bgr = identifyCharacter.find_character(clean_plate, 400)
            char = identifyCharacter.segment_character(thresh, clean_plate)
            lenL = identifyCharacter.length(char)
            if lenL>6 and lenL<9 and x>20:
                #cv2.imwrite("file_1/f"+ str(time.time())+'.jpg',clean_plate)
                cv2.imshow('a', clean_plate)
                X.append(x)
                Y.append(y)
                characters = identifyCharacter.read(char,bgr)
                #print(x, X[-2], y, Y[-2])
                tracking = getDistance(x, y, X[-2], Y[-2])
                List_plate.append(clean_plate)
                cv2.rectangle (frame, (x, y), (x + w, y + h), (0,255,0), 2)
                List = []
                plate_list = ""
                for i in range(len(characters)):
                    tensor = myNetwork.read_tensor_from_image(characters[i],224)
                    label = myNetwork.label_image(tensor)
                    List.append(label)
                if List[2] == '0':
                    List[2] = 'd'
                elif List[2] == '8':
                    List[2] = 'b'
                elif List[2] == '2':
                    List[2] = 'f'
                elif List[2] == '1' or List[2] == '4' :
                    List[2] = 'a'
                elif List[0] == '6':
                    List[0] = '5'
                #print(List)
                for i in range(len(List)):
                    plate_list = plate_list + str(List[i])
                comparison_array.append(plate_list)
                if tracking < 100 and x<X[-2] and y>Y[-2]:
                    cv2.imshow('plate', List_plate[0])
                    len_plate.append(lenL)  
                    length = max(len_plate)
                    if len(comparison_array)>5:
                        comparison_array = [x for x in comparison_array if len(x)==length]
                        for i in range(len(comparison_array)):
                            for j in range(len(comparison_array[i])):
                                change_1D_array.append(comparison_array[i][j])
                        change_2D_array = np.reshape(change_1D_array,(-1,length))
                    final_plate = ""
                    for i in range(change_2D_array.shape[1]):
                        for j in range(change_2D_array.shape[0]):
                            compare_plate.append(change_2D_array[j][i])
                        char_final = most_frequent(compare_plate)
                        final_plate = final_plate + str(char_final)
                        compare_plate.clear()
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,final_plate,(10,75), font, 2,(0,255,0),2,cv2.LINE_AA)
                else:
                    List_plate.clear()
                    len_plate.clear()
                    comparison_array.clear()
                    change_1D_array.clear()
                    change_2D_array.clear()
                
        except:
            pass
    cv2.waitKey(25)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
