from ml.tools.match import Match
import cv2
import numpy as np
import os
import json
import ml.tools.match as match_tools
import ml.tools.file_utils as file_utils
match_db_lst = "MatchDB.csv"
match_lst = match_tools.load_match_list(match_db_filename=match_db_lst, fast=True, absolute_paths=True)
match = match_lst[0]

def rel(point,x0,y0):
    return [point[0]-x0, point[1]-y0]

def intersect_boxes(first,second):
    #first = [x,y,w,h]
    x1,y1,w1,h1 = first
    x2, y2, w2, h2 = second
    return not(x1+w1 < x2 or  x1> x2+w2 or y1+h1 < y2 or y2+h2<y1)

def first_includes_second(first, second):
    x1,y1,w1,h1 = first
    x2, y2, w2, h2 = second
    d = 20
    return x1-d<=x2<=x2+w2<=x1+w1+d and y1-d<=y2<=y2+h2<=y1+h1+d

def create_json_for_file(img_filename, regions):
    dict_file = {"filename": img_filename, "size": 196662, }
    dict_file["regions"] = regions
    dict_file["file_attributes"] = {}
    return dict_file



def cut_contour(contour_resized):
    contour_resized = [[min(255,max(0,point[0])),min(255,max(0,point[1]))] for point in contour_resized]
    return contour_resized




#for alll labels in Players\LabData\Lab2team2Player check if the label is in labdata/contours and extract
match.read_labs_to_players()
match.labels_with_contours_at_frame_dic = {} #sict of frame_num:set_of_labels_with_contour_at_frame
match.labels_with_contours ={}
for label_id, player in list(match.dic_labels_to_players.items())[0:200]:
    label_with_cont=match_tools.LabelWithContours(match, label_id)
    match.labels_with_contours[label_id] = label_with_cont
    contours_in_label = label_with_cont.contours
    for (frame_num,contour) in contours_in_label:
        match.labels_with_contours_at_frame_dic.setdefault(frame_num, set()).add(label_id)


match.dir_contour_images = match.dir_match+"ContoursImages//"
file_utils.try_create_path(match.dir_contour_images)
cap = cv2.VideoCapture(match.video_filename)
print(match.video_filename)
dir_label = match.dir_contour_images
dict_label={}
index=0

dict_all = {}

cv2.namedWindow('image')



ret, frame_img = cap.read()
cv2.imshow("image", frame_img)

def on_mouse_click(event,x,y,flags,param):
    global MouseX,MouseY
    cv2.circle(frame_img,(x,y),100,(255,0,0),thickness=20)
    mouseX = x
    mouseY = y

cv2.setMouseCallback("image",on_mouse_click)
while cap.isOpened():
    #for i in range(20):
    ret, frame_buff = cap.read()
    frame_img[:,:,:] = frame_buff[:,:,:]
    if ret is None:
        continue
    frame_num = cap.get(1)-1
    labels_id_with_contours_at_frame = match.labels_with_contours_at_frame_dic.get(frame_num, None)
    if labels_id_with_contours_at_frame is not None:
        for label_id in labels_id_with_contours_at_frame:
            label = match.labels_with_contours[label_id]

            contour = label.get_contour_at_frame(frame_num)
            contour_p = contour
            contour_rel = contour_p
            #contour_rel = [rel(point, x, y) for point in contour_p]
            #contour_resized = [[int(scale_x * point[0]), int(scale_y * point[1])] for point in contour_rel]
            contour_resized = contour_rel
            contour_rel_np = np.array(contour_resized)

            pts_rel = contour_rel_np.reshape((-1, 1, 2))
            cv2.polylines(frame_img, [pts_rel], True, (0, 255, 0), thickness=2)

            img_filename = "L" + str(label_id) + "F" + str(frame_num) + ".bmp"
            index += 1

    key = cv2.waitKey(500)
    cv2.imshow("image", frame_img)


    #json.dump(dict_label, open(dir_label + "All.json", "wt"))
    if index>=60000:
        break
json.dump(dict_all, open(dir_label + "via_region_data.json", "wt"))


