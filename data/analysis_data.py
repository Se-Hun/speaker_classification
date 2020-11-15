import os
import json
from collections import Counter

def build_data(fns):
    in_fn = fns["input"]

    with open(in_fn, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
        print("Number of source data : {}".format(len(all_data)))

        target_ids = []
        for ex in all_data:
            category = ex["metadata"]["category"]
            if category != "메신저 대화 > 2인 대화":
                id = ex["id"]
                target_ids.append(id)
                print("ID : {}, Category : {}".format(id, category))

        print("Number of targets : {}".format(len(target_ids)))
        print("-------------------------------------------------")

        print()
        print("<<Topic Count>>")
        topics = []
        for ex in all_data:
            id = ex["id"]
            document = ex["document"]
            assert (len(document) == 1), "length of the document list is upper 1 at {}".format(id)

            topic = document[0]["metadata"]["topic"]
            topics.append(topic)

        topic_counter = Counter(topics)
        for k, v in topic_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Topics : {}".format(len(list(topic_counter.keys()))))
        print("-------------------------------------------------")

        print()
        print("<<Speaker Analysis>>")

        ages = []
        occupations = []
        sexes = []
        birthplaces = []
        pricipal_residences = []
        current_residences = []
        devices = []
        keyboards = []
        for ex in all_data:
            id = ex["id"]
            document = ex["document"][0]

            speakers = document["metadata"]["speaker"]
            for speaker in speakers:
                age = speaker["age"]
                occupation = speaker["occupation"]
                sex = speaker["sex"]
                birthplace = speaker["birthplace"]
                pricipal_residence = speaker["pricipal_residence"]
                current_residence = speaker["current_residence"]
                device = speaker["device"]
                keyboard = speaker["keyboard"]

                ages.append(age)
                occupations.append(occupation)
                sexes.append(sex)
                birthplaces.append(birthplace)
                pricipal_residences.append(pricipal_residence)
                current_residences.append(current_residence)
                devices.append(device)
                keyboards.append(keyboard)

        age_counter = Counter(ages)
        for k, v in age_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Ages : {}".format(len(list(age_counter.keys()))))
        print("-------------------------------------------------")

        occupation_counter = Counter(occupations)
        for k, v in occupation_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Occupations : {}".format(len(list(occupation_counter.keys()))))
        print("-------------------------------------------------")

        sex_counter = Counter(sexes)
        for k, v in sex_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Sexs : {}".format(len(list(sex_counter.keys()))))
        print("-------------------------------------------------")

        birthplace_counter = Counter(birthplaces)
        for k, v in birthplace_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Birthplaces : {}".format(len(list(birthplace_counter.keys()))))
        print("-------------------------------------------------")

        device_counter = Counter(devices)
        for k, v in device_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Devices : {}".format(len(list(device_counter.keys()))))
        print("-------------------------------------------------")

        keyboard_counter = Counter(keyboards)
        for k, v in keyboard_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Keyboards : {}".format(len(list(keyboard_counter.keys()))))
        print("-------------------------------------------------")

        print()
        print("<<Speaker Setting Analysis>>")

        relations = []
        intimacies = []
        contact_frequencies = []
        for ex in all_data:
            id = ex["id"]
            document = ex["document"][0]

            setting = document["metadata"]["setting"]
            assert (len(list(setting.keys())) == 3), "number of setting key-value is not 3"

            relations.append(setting["relation"])
            intimacies.append(setting["intimacy"])
            contact_frequencies.append(setting["contact_frequency"])

        relation_counter = Counter(relations)
        for k, v in relation_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Relations : {}".format(len(list(relation_counter.keys()))))
        print("-------------------------------------------------")

        intimacy_counter = Counter(intimacies)
        for k, v in intimacy_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Intimacies : {}".format(len(list(intimacy_counter.keys()))))
        print("-------------------------------------------------")

        contact_frequency_counter = Counter(contact_frequencies)
        for k, v in contact_frequency_counter.items():
            print("{} : {}".format(k, v))
        print("Number of Contact Frequencies : {}".format(len(list(contact_frequency_counter.keys()))))
        print("-------------------------------------------------")

        texts = []
        for ex in all_data:
            utterances = ex["document"][0]["utterance"]
            for utterance in utterances:
                texts.append(utterance)

        print("총 문장 수 : {}".format(len(texts)))

if __name__ == '__main__':
    in_dir = os.path.join("./", "original-all")

    # to_dir = os.path.join("./", "original-all")
    # prepare_dir(to_dir)

    fns = {
        "input": os.path.join(in_dir, "data.json")
        # "output": os.path.join(to_dir, "data.json")
    }

    build_data(fns)