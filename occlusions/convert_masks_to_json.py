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

#for alll labels in Players\LabData\Lab2team2Player check if the label is in labdata/contours and extract
match.read_labs_to_players();
match.dir_contour_images = match.dir_match+"ContoursImages//"
#os.mkdir(match.dir_contour_images)
file_utils.try_create_path(match.dir_contour_images)
cap = cv2.VideoCapture(match.video_filename)
print(match.video_filename)
dir_label = match.dir_contour_images
dict_label={}
index=0
if cap.isOpened():
    print("Is open")
    for label_id, player in match.dic_labels_to_players.items():
        index +=1
        # if index < 200:
        #     continue
        #dir_label = match.dir_contour_images+ str(label_id) +"//"
        file_utils.try_create_path(dir_label)
        #boxes = match.read_label(label_id).boxes
        #contours = match.read_contours(label_id)
        label_with_cont = match_tools.LabelWithContours(match,label_id)
        #dict_label={}

        for contour in label_with_cont.contours:
            regions = []
            frame_no = contour[0]
            contour_p = contour[1]
            #print(contour_p)
            contour_np = np.array(contour_p)
            pts = contour_np.reshape((-1, 1, 2))
            #print(pts)
            cap.set(1, frame_no);
            ret, frame_img = cap.read()

            #print(ret)
            if ret:
                box_line = label_with_cont.boxes[label_with_cont.boxes[:,0]==frame_no][0].astype(np.int)
                if box_line is not None:
                    #print(box_line)
                    x0, y0, w0, h0 = box_line[2:6]
                    hr, wr = box_line[8:10]
                    #cv2.polylines(frame_img, [pts], True, (0, 255, 255), thickness=2)
                    x,y, w, h = match.extract_360cm_by_360cm_box(x0,y0,w0,h0,wr,hr)
                    if (x>0 and y >0 and x+w < 1920 and y+h <1080):
                        print(x,y, w, h )

                        image_player =frame_img[y:y+h, x:x+w]
                        #cv2.rectangle(frame_img, (x, y), (x + w, y + h), color=(0, 255, 0))
                        xr,yr = x - x0, y -y0
                        contour_rel = [rel(point, x, y) for point in contour_p]

                        img_filename = "L"+str(label_id)+"F"+str(frame_no)+".bmp"

                        contour_rel_np = np.array(contour_rel)
                        pts_rel = contour_rel_np.reshape((-1, 1, 2))
                        #cv2.polylines(image_player, [pts_rel], True, (0, 255, 0), thickness=2)



                        #cv2.imwrite(dir_label + img_filename, image_player)
                        #cv2.imshow("small", image_player)

                        image_player_resized = cv2.resize(image_player,dsize=(256,256))
                        scale_x = 256/image_player.shape[0]
                        scale_y = 256 / image_player.shape[1]
                        contour_resized = [[int(scale_x* point[0]), int(scale_y*point[1])] for point in contour_rel]
                        contour_resized_np = np.array(contour_resized)
                        pts_resized = contour_resized_np.reshape((-1, 1, 2))
                        #cv2.polylines(image_player_resized, [pts_resized], True, (0, 0, 255), thickness=2)
                        cv2.imshow("fix", image_player_resized)

                        cv2.imwrite(dir_label +img_filename, image_player_resized)
                        dict_file = {"filename":img_filename,"size":196662, }

                        region = {}

                        shape ={}
                        shape["name"] = "polygon"
                        all_points_x = [point[0] for point in contour_resized]
                        all_points_y =[point[1] for point in contour_resized]
                        shape["all_points_x"] = all_points_x
                        shape["all_points_y"] = all_points_y

                        region["shape_attributes"]=shape
                        region["region_attributes"]= {}
                        regions.append(region)

                        dict_file["regions"] = regions
                        dict_file["file_attributes"]= {}

                        # print(box_line)
                        dict_label[img_filename+str(196662)] = dict_file

                cv2.imshow("big",frame_img)

            #json.dump(dict_label, open(dir_label+ "L"+str(label_id)+".json","wt"))
            key = cv2.waitKey(1);
            if key > 0:
                break
        json.dump(dict_label, open(dir_label + "All.json", "wt"))
        if index==220:
            break



