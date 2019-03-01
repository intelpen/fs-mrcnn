#merges 2 DataSets din dir_db1 and dir_db2
import json
from tools import file_utils
import shutil
import os

dir_db1 = "D://data//ml//mrcnn//dataset_1//"
dir_db2 = "D://data//ml//mrcnn//dataset_2//"
dir_merged = "D://data//ml//mrcnn//dataset_merged//"

file_utils.try_create_path(dir_merged)

db1_json_file  = dir_db1 + "via_region_data.json"
db2_json_file  = dir_db2 + "via_region_data.json"

json1 = json.load(open(db1_json_file))
json2 = json.load(open(db2_json_file))
json_together = json1.copy()
json_together.update(json2)

for filename in os.listdir(dir_db1):
    if filename[-4:] == ".bmp":
        shutil.copy(dir_db1 + filename, dir_merged)

for filename in os.listdir(dir_db2):
    if filename[-4:] == ".bmp":
        shutil.copy(dir_db2 + filename, dir_merged)

json.dump(json_together, open(dir_merged + "via_region_data.json", "wt"))



