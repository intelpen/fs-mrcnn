#copy to path_out only files from path_in also found in path_verif
path_in ="E://Outputs//Outputs 2018-10-28 - FC Barcelona - Real Madrid - Plan Large//LabData//"
path_out = "E://Outputs//Outputs 2018-10-28 - FC Barcelona - Real Madrid - Plan Large//LabData_correct//"
path_verif ="E://Outputs//Outputs 2018-10-28 - FC Barcelona - Real Madrid - Plan Large//Lab2teamVerif//"

import os
import shutil
files_verif = list(os.listdir(path_verif))
labels_verif = [int(filename[3:-4]) for filename in files_verif]

files_in = list(os.listdir(path_in))
labels_in =[int(filename[4:-4]) for filename in files_in if filename[4:-4].isdigit()]

for label_id in labels_in:
    if label_id in labels_verif:
        shutil.copyfile(path_in+"LabB"+str(label_id)+".csv", path_out+"LabB"+str(label_id)+".csv")