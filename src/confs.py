import argparse
from pathlib import Path
import json
import os
import hashlib
import time
from datetime import datetime

slash = "/" #"\\"  # or "/" for linux

base = {
    "templates": {
        "ie": None
    },
    "dataset": None,
}


def create_parser():
    parser = argparse.ArgumentParser(description="IE coordinate catastali con LLMs.", allow_abbrev=False)
    parser.add_argument('--dataset', help='dataset su cui eseguire l\'esperimento')
    parser.add_argument('--model', help='"human", "rules", o un llm"')
    parser.add_argument('--ie_prompt', help='Estrae le entità. Specificare nome prompt.')
    return parser



def parse():
    parser = create_parser()
    args = parser.parse_args()

    if not args.model or not args.dataset or ( (not args.ie_prompt) and (not args.model == "rules") and (not args.model == "human") ) :
        parser.error('model, dataset and at least one prompt should be provided')
    if args.dataset != "test_small" and args.dataset != "test_medium" and args.dataset != "test_full" and args.dataset != "validation_small" and args.dataset != "validation_full" and args.dataset != "monolotto" and args.dataset != "multilotto" :
        print(args.dataset)
        parser.error("dataset can be only test_small, test_medium, test_full, validation_small or validation_full")
    return args
    
    
def build_ouput_file_path(conf):
    output_dir = "results" + slash + conf["dataset"] + slash + conf["model"].replace(":", "_") + "_"
    if conf["templates"]["ie"]:
        output_dir += "_" + conf["templates"]["ie"]
        if ".txt" in  conf["templates"]["ie"]:
            output_dir = output_dir[:-4]

    output_dir += "__V" + conf["version"]
    return output_dir
          

def personalize(args):   
    conf = {}
    conf["templates"] = base["templates"]
    conf["dataset"] = args.dataset
    conf["model"] = args.model
    conf["templates"]["ie"] = args.ie_prompt
    conf["time"] = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    conf["version"] = hashlib.sha256(str(time.time()).encode()).hexdigest()[:4]
    conf["output_dir"] = build_ouput_file_path(conf)
    print(conf["output_dir"])
    return conf



def save(conf):
    f_out = conf["output_dir"] + slash + "conf.json"
    Path(os.path.dirname(f_out)).mkdir(parents=True, exist_ok=True)
    with open(f_out, 'w') as fp:
        json.dump(conf, fp, indent=4)
        

    
def build_conf(args):
    conf = personalize(args)
    save(conf)
    return conf
    

