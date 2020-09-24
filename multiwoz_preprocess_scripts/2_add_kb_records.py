import numpy as np
import json
import random

KB_NUM = 5

dataset = "train"
input_path = "../data/MULTIWOZ2.1/{}.txt".format(dataset)
output_path = "../data/MULTIWOZ2.1/{}_w_kb.txt".format(dataset)

restaurant_kb = []
hotel_kb = []
attraction_kb = []
train_kb = []
hospital_kb = []

##############################
# load database for each task
##############################

with open("../data/MULTIWOZ2.1/restaurant_db.json") as fin:
    restaurant_db = json.load(fin)
    for index, ent in enumerate(restaurant_db):
        # print(ent["name"])
        ent_name = ent["name"]
        # remove danyinhao
        ent_name = ent_name.replace('\'', ' ')
        ent_name_new = []
        for word in ent_name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                ent_name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                ent_name_new.append(word)
            else:
                ent_name_new.append(word)
        ent_name_new = " ".join(ent_name_new)
        restaurant_kb.append(ent_name_new)
    print("load restaurant_db success.")

with open("../data/MULTIWOZ2.1/hotel_db.json") as fin:
    hotel_db = json.load(fin)
    for index, ent in enumerate(hotel_db):
        # print(ent["name"])
        ent_name = ent["name"]
        # remove danyinhao
        ent_name = ent_name.replace('\'', ' ')
        ent_name_new = []
        for word in ent_name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                ent_name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                ent_name_new.append(word)
            else:
                ent_name_new.append(word)
        ent_name_new = " ".join(ent_name_new)
        hotel_kb.append(ent_name_new)
    print("load hotel_db success.")

with open("../data/MULTIWOZ2.1/attraction_db.json") as fin:
    attraction_db = json.load(fin)
    for index, ent in enumerate(attraction_db):
        # print(ent["name"])
        ent_name = ent["name"]
        # remove danyinhao
        ent_name = ent_name.replace('\'', ' ')
        ent_name_new = []
        for word in ent_name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                ent_name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                ent_name_new.append(word)
            else:
                ent_name_new.append(word)
        ent_name_new = " ".join(ent_name_new)
        attraction_kb.append(ent_name_new)
    print("load attraction_db success.")

with open("../data/MULTIWOZ2.1/train_db.json") as fin:
    train_db = json.load(fin)
    for index, ent in enumerate(train_db):
        # print(ent["trainID"])
        ent_name = ent["trainID"]
        # remove danyinhao
        ent_name = ent_name.replace('\'', ' ')
        ent_name_new = []
        for word in ent_name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                ent_name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                ent_name_new.append(word)
            else:
                ent_name_new.append(word)
        ent_name_new = " ".join(ent_name_new)
        train_kb.append(ent_name_new)
    print("load train_db success.")

with open("../data/MULTIWOZ2.1/hospital_db.json") as fin:
    hospital_db = json.load(fin)
    for index, ent in enumerate(hospital_db):
        # print(ent["department"])
        ent_name = ent["department"]
        # remove danyinhao
        ent_name = ent_name.replace('\'', ' ')
        ent_name_new = []
        for word in ent_name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                ent_name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                ent_name_new.append(word)
            else:
                ent_name_new.append(word)
        ent_name_new = " ".join(ent_name_new)
        hospital_kb.append(ent_name_new)
    print("load hospital_db success.")


#####################################################
# find ground-truth KB records for each dialogue
#####################################################

instance_id = 0
instance_kb_info = {}
kb_info = []

with open(input_path) as fin:
    for line in fin:
        line = line.replace("\n", "")
        if line:
            if "#" in line:
                task_type = line.replace("#", "")
                continue
            if "\t" in line:
                nid, line = line.split(" ", 1)
                u, r = line.split("\t")

                if task_type == "restaurant": kb = restaurant_kb
                elif task_type == "hotel": kb = hotel_kb
                elif task_type == "attraction": kb = attraction_kb
                elif task_type == "train": kb = train_kb
                elif task_type == "hospital": kb = hospital_kb

                for ent in kb:
                    if ent in u or ent in r:
                        if ent not in kb_info:
                            kb_info.append(ent)
        else:
            instance_kb_info[instance_id] = kb_info
            instance_id += 1
            kb_info = []
print("find kb info success.")


###############################
# add KB info before dialogues
###############################
# multiwoz_preprocess_scripts KB format
restaurant_kb_processd = {}
hotel_kb_processd = {}
attraction_kb_processed = {}
train_kb_processed = {}
hospital_kb_processed = {}

# Note: KB entities all use lower case characteristics, and concatenate using "_", for address, we also remove ","

# process restaurant data
with open("../data/MULTIWOZ2.1/restaurant_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts name, align with previous stage
        name_t = ent["name"]
        # remove danyinhao
        name_t = name_t.replace('\'', ' ')
        name_new = []
        for word in name_t.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                name_new.append(word)
            else:
                name_new.append(word)
        name_t = " ".join(name_new)
        name = "_".join(name_new)
        # multiwoz_preprocess_scripts address, align with previous stage
        address = ent["address"]
        # remove danyinhao
        address = address.replace('\'', ' ')
        address_new = []
        for word in address.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                address_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                address_new.append(word)
            else:
                address_new.append(word)
        address = "_".join(address_new)

        area = ent["area"]
        area = "_".join(area.lower().split(" "))
        food = ent["food"]
        food = "_".join(food.lower().split(" "))
        location = str(ent["location"][0]) + "_" + str(ent["location"][1])
        if "phone" in ent:
            phone = ent["phone"]
            phone = "_".join(phone.lower().split(" "))
        postcode = ent["postcode"]
        postcode = "_".join(postcode.lower().split(" "))
        price_range = ent["pricerange"]
        price_range = "_".join(price_range.lower().split(" "))
        type = ent["type"]
        type = "_".join(type.lower().split(" "))
        restaurant_kb_processd[name_t] = []
        restaurant_kb_processd[name_t].append(name)
        restaurant_kb_processd[name_t].append(name + " address " + address)
        restaurant_kb_processd[name_t].append(name + " area " + area)
        restaurant_kb_processd[name_t].append(name + " food " + food)
        # restaurant_kb_processd[name_t].append(name + " location " + location)
        if "phone" in ent:
            restaurant_kb_processd[name_t].append(name + " phone " + phone)
        restaurant_kb_processd[name_t].append(name + " postcode " + postcode)
        restaurant_kb_processd[name_t].append(name + " pricerange " + price_range)
        restaurant_kb_processd[name_t].append(name + " type " + type)

# process hotel data
with open("../data/MULTIWOZ2.1/hotel_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts name, align with previous stage
        name_t = ent["name"]
        # remove danyinhao
        name_t = name_t.replace('\'', ' ')
        name_new = []
        for word in name_t.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                name_new.append(word)
            else:
                name_new.append(word)
        name_t = " ".join(name_new)
        name = "_".join(name_new)
        # multiwoz_preprocess_scripts address, align with previous stage
        address = ent["address"]
        # remove danyinhao
        address = address.replace('\'', ' ')
        address_new = []
        for word in address.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                address_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                address_new.append(word)
            else:
                address_new.append(word)
        address = "_".join(address_new)

        area = ent["area"]
        area = "_".join(area.lower().split(" "))
        internet = ent["internet"]
        internet = "_".join(internet.lower().split(" "))
        parking = ent["parking"]
        parking = "_".join(parking.lower().split(" "))
        location = str(ent["location"][0]) + "_" + str(ent["location"][1])
        location = "_".join(location.lower().split(" "))
        phone = ent["phone"]
        phone = "_".join(phone.lower().split(" "))
        postcode = ent["postcode"]
        postcode = "_".join(postcode.lower().split(" "))
        if "single" in ent["price"]:
            single_price = ent["price"]["single"]
            single_price = "_".join(single_price.lower().split(" "))
        if "double" in ent["price"]:
            double_price = ent["price"]["double"]
            double_price = "_".join(double_price.lower().split(" "))
        if "family" in ent["price"]:
            family_price = ent["price"]["family"]
            family_price = "_".join(family_price.lower().split(" "))
        pricerange = ent["pricerange"]
        pricerange = "_".join(pricerange.lower().split(" "))
        stars = ent["stars"]
        stars = "_".join(stars.lower().split(" "))
        if "takesbookings" in ent:
            takesbookings = ent["takesbookings"]
            takesbookings = "_".join(takesbookings.lower().split(" "))
        type = ent["type"]
        type = "_".join(type.lower().split(" "))
        hotel_kb_processd[name_t] = []
        hotel_kb_processd[name_t].append(name)
        hotel_kb_processd[name_t].append(name + " address " + address)
        hotel_kb_processd[name_t].append(name + " area " + area)
        hotel_kb_processd[name_t].append(name + " internet " + internet)
        hotel_kb_processd[name_t].append(name + " parking " + parking)
        # hotel_kb_processd[name_t].append(name + " location " + location)
        hotel_kb_processd[name_t].append(name + " phone " + phone)
        hotel_kb_processd[name_t].append(name + " postcode " + postcode)
        if "single" in ent["price"]:
            hotel_kb_processd[name_t].append(name + " singleprice " + single_price)
        if "double" in ent["price"]:
            hotel_kb_processd[name_t].append(name + " doubleprice " + double_price)
        if "family" in ent["price"]:
            hotel_kb_processd[name_t].append(name + " familyprice " + family_price)
        hotel_kb_processd[name_t].append(name + " pricerange " + pricerange)
        hotel_kb_processd[name_t].append(name + " stars " + stars)
        if "takesbookings" in ent:
            hotel_kb_processd[name_t].append(name + " takesbookings " + takesbookings)
        hotel_kb_processd[name_t].append(name + " type " + type)


# process attraction data
with open("../data/MULTIWOZ2.1/attraction_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts name, align with previous stage
        name_t = ent["name"]
        # remove danyinhao
        name_t = name_t.replace('\'', ' ')
        name_new = []
        for word in name_t.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                name_new.append(word)
            else:
                name_new.append(word)
        name_t = " ".join(name_new)
        name = "_".join(name_new)
        # multiwoz_preprocess_scripts address, align with previous stage
        address = ent["address"]
        # remove danyinhao
        address = address.replace('\'', ' ')
        address_new = []
        for word in address.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                address_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                address_new.append(word)
            else:
                address_new.append(word)
        address = "_".join(address_new)

        area = ent["area"]
        area = "_".join(area.lower().split(" "))
        if "entrance fee" in ent:
            entrance_fee = ent["entrance fee"]
            entrance_fee = "_".join(entrance_fee.lower().split(" "))
        location = str(ent["location"][0]) + "_" + str(ent["location"][1])
        location = "_".join(location.lower().split(" "))
        if "openhours" in ent:
            openhours = ent["openhours"]
            openhours = "_".join(openhours.replace(",", "").lower().split(" "))
        phone = ent["phone"]
        phone = "_".join(phone.lower().split(" "))
        postcode = ent["postcode"]
        postcode = "_".join(postcode.lower().split(" "))
        pricerange = ent["pricerange"]
        pricerange = "_".join(pricerange.lower().split(" "))
        type = ent["type"]
        type = "_".join(type.lower().split(" "))
        # Note: remove data with missing entrance fee and pricerange.
        # if ("entrance fee" in ent and ent["entrance fee"] == "?") or ("pricerange" in ent and ent["pricerange"] == "?"):
        #     continue
        attraction_kb_processed[name_t] = []
        attraction_kb_processed[name_t].append(name)
        attraction_kb_processed[name_t].append(name + " address " + address)
        attraction_kb_processed[name_t].append(name + " area " + area)
        if "entrance fee" in ent and ent["entrance fee"] != "?":
            attraction_kb_processed[name_t].append(name + " entrancefee " + entrance_fee)
        # attraction_kb_processed[name_t].append(name + " location " + location)
        # Note: don't include openhours entity since it's not standard structure!!!
        # if "openhours" in ent:
        #     attraction_kb_processed[name_t].append(name + " openhours " + openhours)
        attraction_kb_processed[name_t].append(name + " phone " + phone)
        attraction_kb_processed[name_t].append(name + " postcode " + postcode)
        if ent["pricerange"] != "?":
            attraction_kb_processed[name_t].append(name + " pricerange " + pricerange)
        attraction_kb_processed[name_t].append(name + " type " + type)


# process train data
with open("../data/MULTIWOZ2.1/train_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        trainid_t = ent["trainID"].lower()
        trainid = ent["trainID"]
        trainid = "_".join(trainid.lower().split(" "))
        day = ent["day"]
        day = "_".join(day.lower().split(" "))
        departure = ent["departure"]
        departure = "_".join(departure.lower().split(" "))
        destination = ent["destination"]
        destination = "_".join(destination.lower().split(" "))
        duration = ent["duration"]
        duration = "_".join(duration.lower().split(" "))
        leaveat = ent["leaveAt"]
        leaveat = "_".join(leaveat.lower().split(" "))
        arriveby = ent["arriveBy"]
        arriveby = "_".join(arriveby.lower().split(" "))
        price = ent["price"]
        price = "_".join(price.lower().split(" "))
        train_kb_processed[trainid_t] = []
        train_kb_processed[trainid_t].append(trainid)
        train_kb_processed[trainid_t].append(trainid + " day " + day)
        train_kb_processed[trainid_t].append(trainid + " departure " + departure)
        train_kb_processed[trainid_t].append(trainid + " destination " + destination)
        train_kb_processed[trainid_t].append(trainid + " duration " + duration)
        train_kb_processed[trainid_t].append(trainid + " leaveat " + leaveat)
        train_kb_processed[trainid_t].append(trainid + " arriveby " + arriveby)
        train_kb_processed[trainid_t].append(trainid + " price " + price)


# process hospital data
with open("../data/MULTIWOZ2.1/hospital_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts department, align with previous stage
        department_t = ent["department"]
        # remove danyinhao
        department_t = department_t.replace('\'', ' ')
        department_new = []
        for word in department_t.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            # if '\'' in word:
            #     word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                department_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                department_new.append(word)
            else:
                department_new.append(word)
        department_t = " ".join(department_new)
        department = "_".join(department_new)

        phone = ent["phone"]
        phone = "_".join(phone.lower().split(" "))
        hospital_kb_processed[department_t] = []
        hospital_kb_processed[department_t].append(department)
        hospital_kb_processed[department_t].append(department + " phone " + phone)


print("process KB data format success.")



############################
# Collect KB head full sets
############################
restaurant_kb_full_sets = [key for key in restaurant_kb_processd]
hotel_kb_full_sets = [key for key in hotel_kb_processd]
attraction_kb_full_sets = [key for key in attraction_kb_processed]
train_kb_full_sets = [key for key in train_kb_processed]
hospital_kb_full_sets = [key for key in hospital_kb_processed]
print("collect KB full sets success.")




##########################
# Output KB
##########################
instance_id = 0
fout = open(output_path, "w")

with open(input_path) as fin:
    for line in fin:
        line = line.replace("\n", "")
        if line:
            if "#" in line:
                task_type = line.replace("#", "")
                fout.write(line + "\n")
                # output KB here
                if task_type == "restaurant":
                    samples = random.sample(restaurant_kb_full_sets, KB_NUM)
                    gold_samples = instance_kb_info[instance_id]
                    if gold_samples:
                        kb_candidates = list(set(samples + gold_samples))
                    else:
                        kb_candidates = samples
                    for key in kb_candidates:
                        ent_list = restaurant_kb_processd[key]
                        for ent in ent_list:
                            fout.write("0 " + ent + "\n")
                elif task_type == "hotel":
                    samples = random.sample(hotel_kb_full_sets, KB_NUM)
                    gold_samples = instance_kb_info[instance_id]
                    if gold_samples:
                        kb_candidates = list(set(samples + gold_samples))
                    else:
                        kb_candidates = samples
                    for key in kb_candidates:
                        ent_list = hotel_kb_processd[key]
                        for ent in ent_list:
                            fout.write("0 " + ent + "\n")
                elif task_type == "attraction":
                    samples = random.sample(attraction_kb_full_sets, KB_NUM)
                    gold_samples = instance_kb_info[instance_id]
                    if gold_samples:
                        kb_candidates = list(set(samples + gold_samples))
                    else:
                        kb_candidates = samples
                    for key in kb_candidates:
                        ent_list = attraction_kb_processed[key]
                        for ent in ent_list:
                            fout.write("0 " + ent + "\n")
                elif task_type == "train":
                    samples = random.sample(train_kb_full_sets, KB_NUM)
                    gold_samples = instance_kb_info[instance_id]
                    if gold_samples:
                        kb_candidates = list(set(samples + gold_samples))
                    else:
                        kb_candidates = samples
                    for key in kb_candidates:
                        ent_list = train_kb_processed[key]
                        for ent in ent_list:
                            fout.write("0 " + ent + "\n")
                elif task_type == "hospital":
                    samples = random.sample(hospital_kb_full_sets, KB_NUM)
                    gold_samples = instance_kb_info[instance_id]
                    if gold_samples:
                        kb_candidates = list(set(samples + gold_samples))
                    else:
                        kb_candidates = samples
                    for key in kb_candidates:
                        ent_list = hospital_kb_processed[key]
                        for ent in ent_list:
                            fout.write("0 " + ent + "\n")
            else:
                fout.write(line + "\n")
        else:
            fout.write("\n")
            instance_id += 1

