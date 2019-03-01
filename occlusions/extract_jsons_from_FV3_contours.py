from ml.tools.match import Match
import cv2
import numpy as np
import os
import json
import ml.tools.match as match_tools
import ml.tools.file_utils as file_utils
match_db_lst = "MatchDB.csv"
match_lst = match_tools.load_match_list(match_db_filename=match_db_lst, fast=True, absolute_paths=True)
match = match_lst[2]

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


def add_json_region(regions, contour_resized):
    region = {}
    shape = {}
    shape["name"] = "polygon"
    all_points_x = [point[0] for point in contour_resized]
    all_points_y = [point[1] for point in contour_resized]
    shape["all_points_x"] = all_points_x
    shape["all_points_y"] = all_points_y

    region["shape_attributes"] = shape
    region["region_attributes"] = {}
    regions.append(region)

def cut_contour(contour_resized):
    contour_resized = [[min(255,max(0,point[0])),min(255,max(0,point[1]))] for point in contour_resized]
    return contour_resized


#for alll labels in Players\LabData\Lab2team2Player check if the label is in labdata/contours and extract
match.read_labs_to_players()
match.labels_with_contours_at_frame_dic = {} #sict of frame_num:set_of_labels_with_contour_at_frame
match.labels_with_contours ={}
for label_id, player in list(match.dic_labels_to_players.items())[0:5000]:
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
while cap.isOpened():
    #for i in range(20):
    ret, frame_img = cap.read()
    if ret is None:
        continue
    frame_num = cap.get(1)-1
    labels_id_with_contours_at_frame = match.labels_with_contours_at_frame_dic.get(frame_num, None)
    if labels_id_with_contours_at_frame is None:
        continue
    for label_id in labels_id_with_contours_at_frame:
        #dir_label = match.dir_contour_images+ str(label_id) +"//"
        #boxes = match.read_label(label_id).boxes
        #contours = match.read_contours(label_id)
        label_with_cont = match.labels_with_contours[label_id]
        neighbour_labels = [label_with_cont]

        box0  = label_with_cont.get_box_px_at_frame(frame_num)


        # extract image
        x0, y0, w0, h0 = label_with_cont.get_box_px_at_frame(frame_num)
        box_line = label_with_cont.boxes[label_with_cont.boxes[:, 0] == frame_num][0].astype(np.int)
        hr, wr = box_line[8:10]  # cv2.polylines(frame_img, [pts], True, (0, 255, 255), thickness=2)
        x, y, w, h = match.extract_x_cm_by_x_cm_box(x0, y0, w0, h0, wr, hr, x=400)
        if (x > 0 and y > 0 and x + w < 1920 and y + h < 1080):
            print(x, y, w, h)

            for label_id_n in labels_id_with_contours_at_frame:
                if label_id_n == label_id:
                    continue
                label_with_cont_n = match.labels_with_contours[label_id_n]
                box1 = label_with_cont_n.get_box_px_at_frame(frame_num)
                if first_includes_second([x,y,w,h], box1):
                    neighbour_labels.append(label_with_cont_n)
            if len(neighbour_labels)<=1:
                print(len(neighbour_labels))
                continue


            image_player = frame_img[y:y + h, x:x + w]
            scale_x = 256 / w
            scale_y = 256 / h
            xr, yr = x - x0, y - y0
            first_image = True
            regions = []
            for label in neighbour_labels:
                contour=label.get_contour_at_frame(frame_num)
                contour_p = contour
                contour_rel = [rel(point, x, y) for point in contour_p]
                contour_resized = [[int(scale_x * point[0]), int(scale_y * point[1])] for point in contour_rel]
                contour_resized = cut_contour(contour_resized)
                contour_rel_np = np.array(contour_resized)
                if first_image is True:
                    first_image = False
                    image_player_resized = cv2.resize(image_player, dsize=(256, 256))
                pts_rel = contour_rel_np.reshape((-1, 1, 2))
                #cv2.polylines(image_player_resized, [pts_rel], True, (0, 255, 0), thickness=2)
                cv2.imshow("fix", image_player_resized)
                add_json_region(regions, contour_resized)

            img_filename = "L" + str(label_id) + "F" + str(frame_num) + ".bmp"
            dict_file = create_json_for_file(img_filename, regions)

            dict_all[img_filename+str(196662)] = dict_file
            cv2.imwrite(dir_label + img_filename, image_player_resized)
            index += 1

    key = cv2.waitKey(10)
    #json.dump(dict_label, open(dir_label + "All.json", "wt"))
    if index>=60000:
        break
json.dump(dict_all, open(dir_label + "via_region_data.json", "wt"))


