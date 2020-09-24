import numpy as np
import json

multiwoz_entities = {"address":[],
                     "area":[],
                     "name":[],
                     "phone":[],
                     "postcode":[],
                     "price":[],
                     "price_attributes":[],
                     "pricerange":[],
                     "stars":[],
                     "type":[],
                     "food_type":[],
                     "entrance_fee":[],
                     "openhours":[],
                     "trainID":[],
                     "time":[],
                     "day":[],
                     "departure":[],
                     "destination":[],
                     "duration":[],
                     "department":[],
                     "boolean":[]}

fout = open("../data/MULTIWOZ2.1/multiwoz_entities.json", "w")
# fout1 = open("../data/MULTIWOZ2.1/multiwoz_entities_hotel.json", "w")
# fout2 = open("../data/MULTIWOZ2.1/multiwoz_entities_restaurant.json", "w")
# fout3 = open("../data/MULTIWOZ2.1/multiwoz_entities_attraction.json", "w")
# fout4 = open("../data/MULTIWOZ2.1/multiwoz_entities_train.json", "w")
# fout5 = open("../data/MULTIWOZ2.1/multiwoz_entities_hospital.json", "w")

# process hotel data
with open("../data/MULTIWOZ2.1/hotel_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts name, align with previous stage
        name = ent["name"]
        name_new = []
        for word in name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            if '\'' in word:
                word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                name_new.append(word)
            else:
                name_new.append(word)
        name = " ".join(name_new)
        # multiwoz_preprocess_scripts address, align with previous stage
        address = ent["address"]
        address_new = []
        for word in address.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            if '\'' in word:
                word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                address_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                address_new.append(word)
            else:
                address_new.append(word)
        address = " ".join(address_new)

        multiwoz_entities["address"].append(address)
        multiwoz_entities["area"].append(ent["area"])
        multiwoz_entities["name"].append(name)
        multiwoz_entities["phone"].append(ent["phone"])
        multiwoz_entities["postcode"].append(ent["postcode"])
        if "single" in ent["price"]:
            multiwoz_entities["price"].append(ent["price"]["single"])
            multiwoz_entities["price_attributes"].append("single")
        if "double" in ent["price"]:
            multiwoz_entities["price"].append(ent["price"]["double"])
            multiwoz_entities["price_attributes"].append("double")
        if "family" in ent["price"]:
            multiwoz_entities["price"].append(ent["price"]["family"])
            multiwoz_entities["price_attributes"].append("family")
        multiwoz_entities["pricerange"].append(ent["pricerange"])
        multiwoz_entities["stars"].append(ent["stars"])
        multiwoz_entities["type"].append(ent["type"])
        multiwoz_entities["boolean"].append(ent["internet"])
        multiwoz_entities["boolean"].append(ent["parking"])
        if "takesbookings" in ent:
            multiwoz_entities["boolean"].append(ent["takesbookings"])
    # remove duplicate entities
    multiwoz_entities["address"] = list(set(multiwoz_entities["address"]))
    multiwoz_entities["area"] = list(set(multiwoz_entities["area"]))
    multiwoz_entities["name"] = list(set(multiwoz_entities["name"]))
    multiwoz_entities["phone"] = list(set(multiwoz_entities["phone"]))
    multiwoz_entities["postcode"] = list(set(multiwoz_entities["postcode"]))
    multiwoz_entities["price"] = list(set(multiwoz_entities["price"]))
    multiwoz_entities["pricerange"] = list(set(multiwoz_entities["pricerange"]))
    multiwoz_entities["price_attributes"] = list(set(multiwoz_entities["price_attributes"]))
    multiwoz_entities["stars"] = list(set(multiwoz_entities["stars"]))
    multiwoz_entities["type"] = list(set(multiwoz_entities["type"]))
    multiwoz_entities["boolean"] = list(set(multiwoz_entities["boolean"]))


# process restaurant data
with open("../data/MULTIWOZ2.1/restaurant_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts name, align with previous stage
        name = ent["name"]
        name_new = []
        for word in name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            if '\'' in word:
                word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                name_new.append(word)
            else:
                name_new.append(word)
        name = " ".join(name_new)
        # multiwoz_preprocess_scripts address, align with previous stage
        address = ent["address"]
        address_new = []
        for word in address.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            if '\'' in word:
                word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                address_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                address_new.append(word)
            else:
                address_new.append(word)
        address = " ".join(address_new)

        multiwoz_entities["address"].append(address)
        multiwoz_entities["area"].append(ent["area"])
        multiwoz_entities["food_type"].append(ent["food"])
        multiwoz_entities["name"].append(name)
        if "phone" in ent:
            multiwoz_entities["phone"].append(ent["phone"])
        multiwoz_entities["postcode"].append(ent["postcode"])
        multiwoz_entities["pricerange"].append(ent["pricerange"])
        multiwoz_entities["type"].append(ent["type"])
    # remove duplicate entities
    multiwoz_entities["address"] = list(set(multiwoz_entities["address"]))
    multiwoz_entities["area"] = list(set(multiwoz_entities["area"]))
    multiwoz_entities["food_type"] = list(set(multiwoz_entities["food_type"]))
    multiwoz_entities["name"] = list(set(multiwoz_entities["name"]))
    multiwoz_entities["phone"] = list(set(multiwoz_entities["phone"]))
    multiwoz_entities["postcode"] = list(set(multiwoz_entities["postcode"]))
    multiwoz_entities["pricerange"] = list(set(multiwoz_entities["pricerange"]))
    multiwoz_entities["type"] = list(set(multiwoz_entities["type"]))


# process attraction data
with open("../data/MULTIWOZ2.1/attraction_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts name, align with previous stage
        name = ent["name"]
        name_new = []
        for word in name.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            if '\'' in word:
                word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                name_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                name_new.append(word)
            else:
                name_new.append(word)
        name = " ".join(name_new)
        # multiwoz_preprocess_scripts address, align with previous stage
        address = ent["address"]
        address_new = []
        for word in address.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            if '\'' in word:
                word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                address_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                address_new.append(word)
            else:
                address_new.append(word)
        address = " ".join(address_new)

        multiwoz_entities["address"].append(address)
        multiwoz_entities["area"].append(ent["area"])
        multiwoz_entities["entrance_fee"].append(ent["entrance fee"])
        multiwoz_entities["name"].append(name)
        multiwoz_entities["openhours"].append(ent["openhours"])
        multiwoz_entities["phone"].append(ent["phone"])
        multiwoz_entities["postcode"].append(ent["postcode"])
        multiwoz_entities["pricerange"].append(ent["pricerange"])
        multiwoz_entities["type"].append(ent["type"])
    # remove duplicate entities
    multiwoz_entities["address"] = list(set(multiwoz_entities["address"]))
    multiwoz_entities["area"] = list(set(multiwoz_entities["area"]))
    multiwoz_entities["entrance_fee"] = list(set(multiwoz_entities["entrance_fee"]))
    multiwoz_entities["name"] = list(set(multiwoz_entities["name"]))
    multiwoz_entities["openhours"] = list(set(multiwoz_entities["openhours"]))
    multiwoz_entities["phone"] = list(set(multiwoz_entities["phone"]))
    multiwoz_entities["postcode"] = list(set(multiwoz_entities["postcode"]))
    multiwoz_entities["pricerange"] = list(set(multiwoz_entities["pricerange"]))
    multiwoz_entities["type"] = list(set(multiwoz_entities["type"]))


# process train data
with open("../data/MULTIWOZ2.1/train_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        multiwoz_entities["trainID"].append(ent["trainID"].lower())
        multiwoz_entities["day"].append(ent["day"])
        multiwoz_entities["departure"].append(ent["departure"])
        multiwoz_entities["destination"].append(ent["destination"])
        multiwoz_entities["duration"].append(ent["duration"])
        multiwoz_entities["time"].append(ent["arriveBy"])
        multiwoz_entities["time"].append(ent["leaveAt"])
        multiwoz_entities["price"].append(ent["price"])
    # remove duplicate entities
    multiwoz_entities["trainID"] = list(set(multiwoz_entities["trainID"]))
    multiwoz_entities["day"] = list(set(multiwoz_entities["day"]))
    multiwoz_entities["departure"] = list(set(multiwoz_entities["departure"]))
    multiwoz_entities["destination"] = list(set(multiwoz_entities["destination"]))
    multiwoz_entities["duration"] = list(set(multiwoz_entities["duration"]))
    multiwoz_entities["time"] = list(set(multiwoz_entities["time"]))
    multiwoz_entities["price"] = list(set(multiwoz_entities["price"]))


# process hospital data
with open("../data/MULTIWOZ2.1/hospital_db.json") as fin:
    data = json.load(fin)
    for ent in data:
        # multiwoz_preprocess_scripts department, align with previous stage
        department = ent["department"]
        department_new = []
        for word in department.split(" "):
            # to lower case
            word = word.lower()
            # remove danyinhao
            if '\'' in word:
                word = word.replace('\'', ' ')
            if word.endswith(','):
                word = word.replace(',', '')
                department_new.append(word)
            elif word.endswith('.'):
                word = word.replace('.', '')
                department_new.append(word)
            else:
                department_new.append(word)
        department = " ".join(department_new)

        multiwoz_entities["department"].append(department)
        multiwoz_entities["phone"].append(ent["phone"])
    # remove duplicate entities
    multiwoz_entities["department"] = list(set(multiwoz_entities["department"]))
    multiwoz_entities["phone"] = list(set(multiwoz_entities["phone"]))

json_out = json.dumps(multiwoz_entities, indent=4, separators=(',', ": "))
fout.write(json_out)

print("success.")

