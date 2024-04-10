import os
import re

classes=['airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge',\
            'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential',\
            'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area',\
            'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain',\
            'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river',\
            'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', \
            'tennis_court', 'terrace', 'thermal_power_station', 'wetland']

def proc_label(file_name):
    with open(file_name,'r') as labelf:
        all_lines = labelf.readlines()

    with open(file_name.split('-')[0]+'.txt','w') as trainf:
        for line in all_lines:
            path = line.strip()
            match = re.search(r'\./([^/]+)/', path)
            label_name = match.group(1)
            if label_name is None:
                raise ValueError
            class_id = classes.index(label_name)
            trainf.write(f"{path} {class_id}\n")

if __name__=="__main__":
    proc_label("train-resisc.txt")