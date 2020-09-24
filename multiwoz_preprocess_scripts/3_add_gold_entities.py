import numpy as np
import json
import pdb

dataset = "train"
input_path = "../data/MULTIWOZ2.1/{}_w_kb.txt".format(dataset)
output_path = "../data/MULTIWOZ2.1/{}_w_kb_w_gold.txt".format(dataset)
fout = open(output_path, "w")

global_entity_set = []
restaurant_entity_set = []
hotel_entity_set = []
attraction_entity_set = []
train_entity_set = []
hospital_entity_set = []

with open("../data/MULTIWOZ2.1/multiwoz_entities.json") as fin:
    global_entities = json.load(fin)
    for key in global_entities:
        global_entity_set = global_entity_set + global_entities[key]

with open("../data/MULTIWOZ2.1/multiwoz_entities_restaurant.json") as fin:
    global_entities = json.load(fin)
    for key in global_entities:
        restaurant_entity_set = restaurant_entity_set + global_entities[key]

with open("../data/MULTIWOZ2.1/multiwoz_entities_hotel.json") as fin:
    global_entities = json.load(fin)
    for key in global_entities:
        hotel_entity_set = hotel_entity_set + global_entities[key]

with open("../data/MULTIWOZ2.1/multiwoz_entities_attraction.json") as fin:
    global_entities = json.load(fin)
    for key in global_entities:
        attraction_entity_set = attraction_entity_set + global_entities[key]

with open("../data/MULTIWOZ2.1/multiwoz_entities_train.json") as fin:
    global_entities = json.load(fin)
    for key in global_entities:
        train_entity_set = train_entity_set + global_entities[key]

with open("../data/MULTIWOZ2.1/multiwoz_entities_hospital.json") as fin:
    global_entities = json.load(fin)
    for key in global_entities:
        hospital_entity_set = hospital_entity_set + global_entities[key]


with open(input_path) as fin:
    for line in fin:
        gold_ent = []
        line = line.replace("\n", "")
        if line:
            if line.startswith("#"):
                task_type = line.replace("#", "")
                fout.write(line + "\n")
                continue
            if "\t" in line:
                nid, line = line.split(" ", 1)
                u, r = line.split("\t")
                if task_type == "restaurant":
                    for ent in restaurant_entity_set:
                        if ent in u:
                            if ent.split(" ")[0] in u.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                u = u.replace(ent, ent_new)
                        if ent in r:
                            if ent.split(" ")[0] in r.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                r = r.replace(ent, ent_new)
                                gold_ent.append(ent_new)
                elif task_type == "hotel":
                    for ent in hotel_entity_set:
                        if ent in u:
                            if ent.split(" ")[0] in u.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                u = u.replace(ent, ent_new)
                        if ent in r:
                            if ent.split(" ")[0] in r.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                r = r.replace(ent, ent_new)
                                gold_ent.append(ent_new)
                elif task_type == "attraction":
                    for ent in attraction_entity_set:
                        if ent in u:
                            if ent.split(" ")[0] in u.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                u = u.replace(ent, ent_new)
                        if ent in r:
                            if ent.split(" ")[0] in r.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                r = r.replace(ent, ent_new)
                                gold_ent.append(ent_new)
                elif task_type == "train":
                    for ent in train_entity_set:
                        if ent in u:
                            if ent.split(" ")[0] in u.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                u = u.replace(ent, ent_new)
                        if ent in r:
                            if ent.split(" ")[0] in r.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                r = r.replace(ent, ent_new)
                                gold_ent.append(ent_new)
                elif task_type == "hospital":
                    for ent in hospital_entity_set:
                        if ent in u:
                            if ent.split(" ")[0] in u.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                u = u.replace(ent, ent_new)
                        if ent in r:
                            if ent.split(" ")[0] in r.split(" "):  # remove miss match badcase.
                                ent_new = "_".join(ent.split(" "))
                                r = r.replace(ent, ent_new)
                                gold_ent.append(ent_new)
                gold_ent = list(set(gold_ent))
                fout.write(nid + " " + u + "\t" + r + "\t" + str(gold_ent) + "\n")
            else:
                fout.write(line + "\n")
        else:
            fout.write("\n")

print("success.")

