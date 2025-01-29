
tipi = ["tipo_immobile", "lotto", "foglio", "particella", "sub"]

def change_format(dataset):
    for el in dataset:
        new_labels = list()
        for label in el["labels"]:
            nome_lotto = label["nome_lotto"]
            for fabbricato in label["fabbricati"]:
                fabbricato["tipo_immobile"] = "fabbricato"
                fabbricato["lotto"] = nome_lotto
                new_labels.append(fabbricato)
            for terreno in label["terreni"]:
                terreno["tipo_immobile"] = "terreno"
                terreno["lotto"] = nome_lotto
                new_labels.append(terreno)
        el["labels"] = clean_immobili(new_labels)
    return dataset



# prende in input le predizioni/gt per un solo documento
def normalize(labels):
    for label in labels:
        for key, value in label.items():
            label[key] = str(value)
            if key == "lotto":
                label[key] = remove_punc_beginning_end(value).lower()

    return labels



def remove_punc_beginning_end(entity):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    try:
        if entity[0] in punctuations:
            entity = entity[1:]
        if entity[-1] in punctuations:
            entity = entity[:-1]
        return entity
    except:
        return entity



def clean_immobili(items):
    clean_items = []
    for item in items:
        clean_item = {}
        for field in item.keys():
            if field.lower() in tipi:
                clean_item[field.lower()] = item[field]
        clean_items.append(clean_item)

    return clean_items


def gt_preprocessing(dataset):
    dataset = change_format(dataset)
    for item in dataset:
        item["labels"] = normalize(item["labels"])
    return dataset
