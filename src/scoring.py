
import json
from pathlib import Path
import os


stats_template = {"match": 0,
         "tot_y": 0,
         "tot_pred": 0,
         "new": 0,
         "malformed_json_error": 0,
         "empty_json": 0,
         "generic_error": 0,
         "too_many_tokens_error": 0,
         "no_response_error": 0,
         }

stats_short_template = {"match": 0,
         "tot_y": 0,
         "tot_pred": 0,
         "new": 0,
         "empty_json": 0
         }



def final_stats(stats, output_file):
    stats["precision"] = stats["match"] / (stats["match"] + stats["new"])
    stats["recall"] = stats["match"] / stats["tot_y"]
    stats["f1"] = 2 * (( stats["precision"] * stats["recall"]) / (stats["precision"] + stats["recall"]))
    save_stats(stats, output_file)
    return stats


def global_entities_scoring(Y, preds, output_file, is_noerror_mode=False):
    if is_noerror_mode:
        stats = stats_short_template.copy()
    else:
        stats = stats_template.copy()
    for y, pred in zip(Y, preds): # for each document
        stats["tot_y"] += len(y)

        if isinstance(pred, str) and "error" in pred:
            if pred == "malformed_json_error":
                stats["malformed_json_error"] +=1
            elif pred == "too_many_tokens_error":
                stats["too_many_tokens_error"] +=  1
            elif pred == "generic_error":
                stats["generic_error"] += 1
            elif pred == "no_response_error":
                stats["no_response_error"] += 1 
        else:
            for item_y in y:
                if item_y in pred:
                    stats["match"] += 1

            stats["tot_pred"] += len(pred)

            if len(pred) == 0:
                stats["empty_json"] += 1
            for item_pred in pred:
                if not item_pred in y:
                    stats["new"] += 1

    return final_stats(stats, output_file)



def save_stats(stats, output_f):
    output_dir = Path(output_f).parent.absolute()
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
        
    with open(output_f, "w", encoding='utf-8') as json_file:
        json.dump(stats, json_file, indent=4)
        
       
        
def save(content, output_dir, f_name):
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
    
    if "json" in f_name:
        with open(output_dir + "/" + f_name, "w", encoding='utf-8') as json_file:
            json.dump(content, json_file, indent=4)
    else:
        with open(output_dir + "/" + f_name, "w", encoding='utf-8') as text_file:
            text_file.write(content)