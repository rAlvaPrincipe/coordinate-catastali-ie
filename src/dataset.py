import json
import random
from preprocessing import gt_preprocessing
from normalizer import Normalizer

class Dataset:

    def __init__(self):
        monolotto_ds = self.read_dataset("data/sanitized_monolotto.json")
        multilotto_ds = self.read_dataset("data/sanitized_multilotto.json")

        monolotto_ds = gt_preprocessing(monolotto_ds)
        multilotto_ds = gt_preprocessing(multilotto_ds)

        val_small_temp, val_full_temp, test_small_temp, test_medium_temp, test_full_temp = self.split(monolotto_ds)
        val_small_temp2, val_full_temp2, test_small_temp2, test_medium_temp2, test_full_temp2 = self.split(multilotto_ds)
        self.val_small_set = val_small_temp + val_small_temp2
        self.val_full_set = val_full_temp + val_full_temp2
        self.test_small_set = test_small_temp + test_small_temp2
        self.test_medium_set = test_medium_temp + test_medium_temp2
        self.test_full_set = test_full_temp + test_full_temp2
        self.monolotto = monolotto_ds
        self.multilotto = multilotto_ds



    def get_dataset(self, split):
        X, Y, ids = [], [], []
        if split == "validation_small":
            X = [el["text"] for el in self.val_small_set]
            Y = [el["labels"] for el in self.val_small_set]
            ids = [el["id"] for el in self.val_small_set]
        elif split == "validation_full":
            X = [el["text"] for el in self.val_full_set]
            Y = [el["labels"] for el in self.val_full_set]
            ids = [el["id"] for el in self.val_full_set]
        elif split == "test_small":
            X = [el["text"] for el in self.test_small_set]
            Y = [el["labels"] for el in self.test_small_set]
            ids = [el["id"] for el in self.test_small_set]
        elif split == "test_medium":
            X = [el["text"] for el in self.test_medium_set]
            Y = [el["labels"] for el in self.test_medium_set]
            ids = [el["id"] for el in self.test_medium_set]
        elif split == "test_full":
            X = [el["text"] for el in self.test_full_set]
            Y = [el["labels"] for el in self.test_full_set]
            ids = [el["id"] for el in self.test_full_set]
        elif split == "monolotto":
            X = [el["text"] for el in self.monolotto]
            Y = [el["labels"] for el in self.monolotto]
            ids = [el["id"] for el in self.monolotto]
        elif split == "multilotto":
            X = [el["text"] for el in self.multilotto]
            Y = [el["labels"] for el in self.multilotto]
            ids = [el["id"] for el in self.multilotto]

        normalizer = Normalizer()
        for y in Y:
            y = normalizer.normalize_immobili(y)
        return X, Y, ids

    def stats(self, dataset):
        terreni_count = 0
        fabbricati_count = 0
        lotti_count = 0
        multilotto_count = 0
        monolotto_count = 0
        for avviso in dataset:
            for lotto in avviso["labels"]:
                terreni_count += len(lotto["terreni"])
                fabbricati_count += len(lotto["fabbricati"])
                lotti_count += 1

            if len(avviso["labels"]) > 1:
                multilotto_count += 1
            else:
                monolotto_count += 1


        print("# tot avvisi: " + str(len(dataset)))
        print("# avvisi monolotto: " + str(monolotto_count))
        print("# avvisi multilotto: " + str(multilotto_count))
        print("# lotti: " + str(lotti_count))
        print("# terreni: " + str(terreni_count))
        print("# fabbricati: " + str(fabbricati_count))


    #per un dataset di 200 items gli split sono così organizati:
    # validation: [0,25] [0,50] test: [50:75] [50:100] [50:200]
    def split(self, dataset):
        random.seed(1992)
        random.shuffle(dataset)
        cut_point = int(len(dataset)/8)
        val_small_set = dataset[:cut_point]
        val_full_set = dataset[:cut_point*2]
        test_small_set = dataset[cut_point*2:cut_point*3]
        test_medium_set = dataset[cut_point*2:cut_point*4]
        test_full_set = dataset[cut_point*2:]
        return val_small_set, val_full_set, test_small_set, test_medium_set, test_full_set


    def read_dataset(self, file_path):
        if "monolotto" in file_path:
            type = "monolotto"
        else:
            type = "multilotto"

        with open(file_path, 'r', encoding='utf-8') as data_file:
            json_data = data_file.read()
        data = json.loads(json_data)

        dataset = list()
        for el in data:
            text = el["plain_text"]
            lotti = list()
            for lotto in el["validation_json"]["lotti"]:
                terreni = list()
                fabbricati = list()
                for terreno in lotto["terreni"]:
                    terreni.append({"comune": terreno["comune"], "foglio": terreno["foglio"], "particella": terreno["particella"]})

                for fabbricato in lotto["fabbricati"]:
                    fabbricato_data = {"comune": fabbricato["comune"], "foglio": fabbricato["foglio"], "particella": fabbricato["particella"]}
                    if "sub" in fabbricato.keys():
                        fabbricato_data["sub"] = fabbricato["sub"]
                    fabbricati.append(fabbricato_data)

                lotti.append({"nome_lotto": lotto["nomeLotto"], "terreni": terreni, "fabbricati": fabbricati})
            dataset.append({"id": el["validation_json"]["avvisoDocumentId"], "text": text, "labels": lotti, "type": type})
        return dataset



    def get_ft_dataset(self):
        from datasets import Dataset as ds
        data = {"input": [], "output": [], "instruction": []}

        with open("prompt_templates/instructions.txt", 'r', encoding='utf-8') as file:
            instruction = file.read()

        for instance in self.val_full_set:
            data["input"].append(instance["text"])
            data["instruction"].append(instruction)
            output = "\"immobili\": " + json.dumps(instance["labels"], indent=3)
            data["output"].append(output)

        return ds.from_dict(data)


    def get_human_annotations(self):
        monolotto_human_ds = self.read_dataset("data/Renzo_monolotto.json")
        multilotto_human_ds = self.read_dataset("data/Renzo_multilotto.json")
        monolotto_human_ds = gt_preprocessing(monolotto_human_ds)
        multilotto_human_ds = gt_preprocessing(multilotto_human_ds)
        human_ds = monolotto_human_ds + multilotto_human_ds
        return human_ds