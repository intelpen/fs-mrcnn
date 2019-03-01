from ml.tools.match import Match
from ml.tools.match import load_match_list
import pandas as pd
import cv2


def extract_occlusion_images():
    pass

match_lst = load_match_list("MatchDB.csv", fast=True,  absolute_paths = True)
match = match_lst[0]



cap = cv2.VideoCapture(match.video_filename)

if not cap.isOpened():
    print("Not opened")
df_images_list = pd.read_csv("D://images_mrcnn.csv", header= None)
for index, line in df_images_list.iterrows():
    print(line)
    frame, x,y, w, h = line
    print(frame,x,y,w,h)
    cap.set(1,frame)
    ret, image = cap.read()
    w_new = h_new = int(1.2*max(h,w))
    dx = (w_new-w)//2
    dy =(h_new-h)//2
    image = image[max(0,y-dy):min(image.shape[0],y+h_new), max(0,x-dx):min(image.shape[1],x+w_new)]
    cv2.imshow("Images",image)
    cv2.imwrite("D://images_clean//"+str(frame)+"_"+str(index)+".bmp", image)
    cv2.waitKey(1)





