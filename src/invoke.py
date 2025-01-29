import argparse
import json
import pprint
from ie import extract

text = """
    Questo è un doc di prova. Comune di Milano lotto 4 foglio 2 particella 2
"""
llm =  "anthropic.claude-3-haiku-20240307-v1:0"

def handler(event, context):
    if "body" in event:
        event = json.loads(event["body"])
    llm = event['llm']
    doc = event['doc']
    return extract(llm, doc)


def invoke_local():
    out = handler({"llm": llm, "doc": text}, None)
    pprint.pprint(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoke LLM extraction with text input.")
    parser.add_argument("--llm", required=True, help="Specify the LLM model")
    parser.add_argument("--text", required=True, help="Provide the input text")

    args = parser.parse_args()
    invoke_local(args.llm, args.text)


