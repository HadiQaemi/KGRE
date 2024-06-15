from datasets import load_dataset

import argparse
import yaml

def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

def prepare_load_data(src, test_src, desc):
    with open(os.path.join(data_dir,"data","nyt",src), 'r') as j:
        contents = json.loads(j.read())

    with open(os.path.join(data_dir,"data","nyt",test_src), 'r') as j:
        contents_test = json.loads(j.read())

    with open(os.path.join(data_dir,"data","nyt",desc), 'w', newline='') as csvfile:
        fieldnames = ['text', 'relations']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        counter = 0
        for item in contents:
            counter = counter + 1
            tags = "*"
            for ii in item['relation_list']:
                relations = ii['predicate'].split('/')
                predicate = relations[len(relations) - 1]
                tags = tags + " " + predicate.replace('/', '_')
                # tags = tags + " " + ii['predicate'].replace('/', '_')
                writer.writerow({
                'text': item['text'],
                'relations': tags.replace("* ", "")
                })
            for item in contents_test:
                tags = "*"
                for ii in item['triple_list']:
                    relations = ii[1].split('/')
                    predicate = relations[len(relations) - 1]
                    tags = tags + " " + predicate.replace('/', '_')
                    counter = counter + 1
                writer.writerow({
                    'text': item['text'],
                    'relations': tags.replace("* ", "")
                    })
