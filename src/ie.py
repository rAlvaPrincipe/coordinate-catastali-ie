from dataset import Dataset
from preprocessing import normalize, clean_immobili
from scoring import global_entities_scoring, save
import json
from confs import parse, build_conf
import os
from llms import Llms
import requests
from normalizer import Normalizer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

llms = Llms()

def build_prompt(template_f, text):
    if os.path.isdir("prompt_templates"):
        template_f = "prompt_templates/" + template_f
    else:
        template_f = "../prompt_templates/" + template_f

    if ".txt" not in template_f:
        template_f += ".txt"
    with open(template_f, "r", encoding='utf-8') as f:
        template = f.read()

    if "llama" in template_f:
        return template + "\n" + text + "\n\n### Response:\n\nRisposta per Input 2:"
    else:
        return template + "\n" + text + "\n\nRisposta per Input 2:"


def res2dict(pred):
    pred = pred.replace("\_", "_")
    if isinstance(pred, str) and " error " in pred:
        return pred
    else:

        try:
            if "[" in pred:
                immobili = json.loads(pred[pred.index("[")  :pred.index("]")+1])
            else:
                immobili = "malformed_json_error"
        except:
            immobili =  "malformed_json_error"
        return immobili



def evaluate_ie(llm, X, Y, ids, output_dir, template_f):
    preds, preds_noerror, Y_noerror = [], [], []
    for id, x, y in zip(ids, X, Y):
        res, prompt, pred = inference_ie(llm, x, template_f)

        # mantieni solo i tipi di interesse
        if "error" not in pred:
            pred = clean_immobili(pred)
            pred = normalize(pred)
        preds.append(pred)

        if "error" not in pred:
            Y_noerror.append(y)
            preds_noerror.append(pred)

        new_ent = diff(y, pred, True)
        missed_ent = diff(y, pred, False)

        save(prompt, output_dir + "/logs/" + str(id) + "/ie/", "prompt_ie.txt")
        save(y, output_dir + "/logs/" + str(id) + "/ie/", "gt_ner.json")
        save(pred, output_dir + "/logs/" + str(id) + "/ie/", "pred_ner.json")
        save(new_ent, output_dir + "/logs/" + str(id) + "/ie/", "new_ner.json")
        save(missed_ent, output_dir + "/logs/" + str(id) + "/ie/", "missed_ner.json")
        save(res, output_dir + "/raw/" + str(id) + "/", "ie.txt")
        save(x, output_dir + "/logs/" + str(id) + "/", "document.txt")

    return preds, preds_noerror, Y_noerror


def evaluate_ie_rules(X, Y, ids, output_dir):
    preds = []
    for id, x, y in zip(ids, X, Y):
        res, pred = inference_ie_rules(x)

        # mantieni solo i tipi di interesse
        pred = clean_immobili(pred)
        pred = normalize(pred)
        preds.append(pred)

        new_ent = diff(y, pred, True)
        missed_ent = diff(y, pred, False)

        save(y, output_dir + "/logs/" + str(id) + "/ie/", "gt_ner.json")
        save(pred, output_dir + "/logs/" + str(id) + "/ie/", "pred_ner.json")
        save(new_ent, output_dir + "/logs/" + str(id) + "/ie/", "new_ner.json")
        save(missed_ent, output_dir + "/logs/" + str(id) + "/ie/", "missed_ner.json")
        save(str(res), output_dir + "/raw/" + str(id) + "/", "ie.txt")
        save(x, output_dir + "/logs/" + str(id) + "/", "document.txt")
    return preds


def evaluate_ie_human(X, Y, ids, human_annotations, output_dir):
    preds = []
    normalizer = Normalizer()
    for id, x, y in zip(ids, X, Y):
        pred = human_annotations[id]["labels"]

        # mantieni solo i tipi di interesse
        pred = clean_immobili(pred)
        pred = normalizer.normalize_immobili(pred)
        preds.append(pred)

        new_ent = diff(y, pred, True)
        missed_ent = diff(y, pred, False)

        save(y, output_dir + "/logs/" + str(id) + "/ie/", "gt_ner.json")
        save(pred, output_dir + "/logs/" + str(id) + "/ie/", "pred_ner.json")
        save(new_ent, output_dir + "/logs/" + str(id) + "/ie/", "new_ner.json")
        save(missed_ent, output_dir + "/logs/" + str(id) + "/ie/", "missed_ner.json")
        save(x, output_dir + "/logs/" + str(id) + "/", "document.txt")
    return preds

def inference_ie_rules(doc):
    response = requests.get('https://z47mwtcx5kgux43goitatbbmze0shrrk.lambda-url.eu-central-1.on.aws/',json={'doc': doc} )
    coordinate_out = []
    try:
        response = response.json()
        for lotto in response["lotti"]:
            comune = lotto["comune"]["nome"]
            nome_lotto = lotto["lotto"]
            for tupla in lotto["tuple"]:
                tmp = { "lotto": nome_lotto, "tipo_immobile": tupla["type"], "comune": comune, "foglio": tupla["foglio"], "particella": tupla["particella"]}
                if "sub" in tupla.keys():
                    tmp["sub"] = tupla["sub"]
                coordinate_out.append(tmp)
    except Exception as e:
        print(response)

    normalizer = Normalizer()
    coordinate_out = normalizer.normalize_immobili(coordinate_out)
    return response, coordinate_out



def inference_ie(llm, doc, template_f):
    prompt = build_prompt(template_f, doc)
    res = llms.ask(prompt, llm)
    pred_immobili = res2dict(res)

    if "error" not in pred_immobili:
        normalizer = Normalizer()
        pred_immobili = normalizer.normalize_immobili(pred_immobili)
    return res, prompt, pred_immobili





# questo metodo calcola le differenze tra le predizioni e la ground truth. il flag "is_new" serve per indicare se ritornare i documenti in più o quelli mancati.
def diff(y, pred, is_new):
    difference = list()
    if is_new:
        for el_pred in pred:
            exist = False
            for el_y in y:
                if el_pred == el_y:
                    exist = True
            if not exist:
                difference.append(el_pred)
    if not is_new:
        for el_y in y:
            exist = False
            for el_pred in pred:
                if el_pred == el_y:
                    exist = True
            if not exist:
                difference.append(el_y)
    return difference



def extract(llm, doc):
    out = {}
    if "llama" in llm:
        template = "ie-v3-claude"
    else:
        template = "ie-v2-llama"

    res, prompt, pred = inference_ie(llm, doc, template)
    out["ie"] = {"immobili": pred}
    return out


if __name__ == "__main__":
    args = parse()
    conf = build_conf(args)
    dataset = Dataset()

    if conf["model"] == "llama3-local":
        llms.load_local_llm(False)
    elif conf["model"] == "llama3-local-ft":
        llms.load_local_llm(True)


    X, Y, ids = dataset.get_dataset(conf["dataset"])
    if conf["model"] == "rules":
        preds = evaluate_ie_rules(X, Y, ids, conf["output_dir"])
        global_entities_scoring(Y, preds, conf["output_dir"] + "/metrics/global/global_stats.json" )
    elif conf["model"] == "human":
        human_annotations = dataset.get_human_annotations()
        id2human_annotation = dict()
        for el in human_annotations:
            id2human_annotation[el["id"]] = el
        preds = evaluate_ie_human(X, Y, ids, id2human_annotation, conf["output_dir"])
        global_entities_scoring(Y, preds, conf["output_dir"] + "/metrics/global/global_stats.json" )
    else:
        preds, preds_noerror, Y_noerror = evaluate_ie(conf["model"], X, Y, ids, conf["output_dir"], conf["templates"]["ie"])
        global_entities_scoring(Y, preds, conf["output_dir"] + "/metrics/global/global_stats.json" )
        global_entities_scoring(Y_noerror, preds_noerror, conf["output_dir"] + "/metrics/global/global_stats_noerror.json", True)

