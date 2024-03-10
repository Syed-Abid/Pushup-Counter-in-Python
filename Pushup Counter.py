import cv2
import mediapipe as md

md_drawing = md.solutions.drawing_utils   #these two lines are basically functions from mediapipe package
md_pose = md.solutions.pose               # and are used to draw lines on your body which you see on the screen.

count = 0  #here we will store the number of pushups done.
position = None

cap = cv2.VideoCapture(0)  #this will take the video input from webcam. The zero in the bracket indicates
                           #that we need the input from the webcam.
with md_pose.Pose(
    min_detection_confidence=0.7,  #it is the accuracy of teh detection of your features.
    min_tracking_confidence=0.7) as pose:    #it tracks the movement of your body.

#this while loop will run till the video is running on the screen as we have used the function 'isOpened()'.    
    while cap.isOpened():     
        success, image = cap.read()     #the image variable will read the video getting captured by webcam
        if not success:                 #and success variable will check that whether we are getting the video
            print("Empty Camera")       #input or not.
            break
            
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) #it's the colorclass we will be using for
        result = pose.process(image)                                #our lines that will appear on the screen. 'cv2.flip' will flip the image completely 

        inList = []  #this list will contain all the coordiates of all the points.

        if result.pose_landmarks:   #this condition will return 'True' when there will be a body present on the screen.
            md_drawing.draw_landmarks(
                image, result.pose_landmarks, md_pose.POSE_CONNECTIONS  #to draw lines on the body we are using 'draw_landmarks' function in which the parameters means this:
            )  #the image parameter is the video where these lines will be displayed, result.pose_landmarks has all the points and landmarks in it, POSE_CONNECTIONS has all the lines which will connects our poses.
            for id, im in enumerate(result.pose_landmarks.landmark): #'result.pose_landmarks.landmark' will create a long list of landmarks on which we will iterate and then we will get two values
                h, w, _ = image.shape  #'shape' function will give us the length and width of our video.
                X, Y = int(im.x * w), int(im.y * h) #this will provide us the exact co-ordinates.
                inList.append([id, X, Y]) #this will append those coordinates with their id in the list.

#so, to do a complete pushup, we must go down in which our shoulders will be below our elbows and then we have to comeup
#in which our shoulders would be above our elbow. This whole logic isbeing applied in the conditions below.
# 11 & 12 are shoulder points and 13 & 14 are elbow points.                 
        if len(inList) != 0: 
            if inList[12][2] >= inList[14][2] and inList[11][2] >= inList[13][2]:
                position = "down"
            if inList[12][2] <= inList[14][2] and inList[11][2] <= inList[13][2] and position == "down":
                position = "up"
                count += 1
                print(count)

        cv2.imshow("Pushup Counter", cv2.flip(image, 1))  #the 'imshow' method shows the output to the user.
        key = cv2.waitKey(1) #this is the exit condition
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
