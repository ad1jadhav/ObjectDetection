
import cv2
import pyttsx3

engine = pyttsx3.init()


def AI_speak(command):
    engine.say(command)
    engine.runAndWait()  
    
    
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
classNames = []
classFile = 'coco.names'
    
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n') 
        
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

notification_count = 0

while True:
    success, img = cap.read()
    if img is None:
        print("Wrong webcam selection.")
    else:
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
            
        try:
            if len(classIds !=0):
                    
                for classID, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    try:
                        label = classNames[classID - 1].upper()
                        confidence = round(confidence * 100, 2)
                        if label == 'PERSON' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("Hello Mr.Aditya") 
                                notification_count += 1
                        elif notification_count > 0:
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0  
                                
                        if label == 'CAR' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("This is a CAR") 
                                notification_count += 1
                        elif notification_count > 0:
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0 
                                
                        if label == 'DOG' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("This is a DOG") 
                                notification_count += 1
                        elif notification_count > 0:
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0  
                                
                        if label == 'CAT' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("This is a CAT") 
                                notification_count += 1
                        elif notification_count > 0:    
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0   
                                
                        if label == 'TOOTHBRUSH' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("This is a Toothbrush") 
                                notification_count += 1
                        elif notification_count > 0:    
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0 
                                
                        if label == 'PIZZA' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("This is a pizza") 
                                notification_count += 1
                        elif notification_count > 0:    
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0    
                                
                        if label == 'BOTTLE' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("This is a water bottle")  
                                notification_count += 1
                        elif notification_count > 0:    
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0    
                                
                        if label == 'BANANA' and confidence > 70:
                            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                            cv2.putText(img, label, (box[0] + 10, box[1] + 30),
                                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) 
                            print(label + ": " + str(confidence))
                            if notification_count == 0:
                                AI_speak("This is a banana")  
                                notification_count += 1
                        elif notification_count > 0:    
                            pass #AI_speak("This is not a SCISSORS")   
                            notification_count = 0    
                                    
                    except IndexError:
                        pass
            else:
                pass
        except TypeError:
            pass
            
        cv2.imshow("Output",img) 
        cv2.waitKey(1)                            