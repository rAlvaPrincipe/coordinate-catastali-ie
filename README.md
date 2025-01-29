# Cadastral Coordinates: Information Extraction with LLMs

This project focuses on **automated extraction of cadastral coordinates** from real estate documents, particularly auction notices, using **Large Language Models (LLMs)**. The extracted information includes **lot, municipality (full name), map sheet (foglio), parcel (particella), subunit (sub), and property type**.

## Invocation

- Use the `invoke.py` script.
- Specify the LLM model to use.
- Provide the text to be processed.

Example:

```sh
python invoke.py --llm "anthropic.claude-3-5-sonnet-20240620-v1:0" --text "This is a test document. Municipality of Milan, lot 2, map sheet 2, parcel 5, subunit 45."
```

### Expected Output:

```json
{
  "statusCode": 200,
  "body": {
    "ie": {
      "immobili": [
        {
          "comune": "Milano",
          "foglio": "2",
          "particella": "5",
          "sub": "45",
          "tipo_immobile": "fabbricato",
          "lotto": "lotto 2"
        }
      ]
    }
  }
}
```

## Fine-Tuning

Modify fine-tuning parameters as needed:

```sh
..\venv\Scripts\python src\llms_ft.py
```

## Evaluation

The evaluation process compares performance across different models.

### Datasets

- **monolotto**: ~150 samples
- **multilotto**: ~50 samples
- **test_small**: 25 samples
- **test_medium**: 50 samples (includes `test_small`)
- **test_full**: 150 samples (includes `test_medium`)
- **validation_small**: 25 samples
- **validation_full**: 50 samples (includes `validation_small`)


### Prompts

#### Claude LLMs:
- `ie-v1-claude`
- `ie-v2-claude`
- `ie-v3-claude`

#### Llama LLMs:
- `ie-v1-llama`
- `ie-v2-llama`

### Models

- **Human** (compares human annotations with corrected references)
- **Rules** (grammar-based extractor on AWS Lambda)
- **Anthropic Claude**:
  - `claude-instant-v1`
  - `claude-v2`
  - `claude-v2:1`
  - `claude-3-sonnet-20240229-v1:0`
  - `claude-3-haiku-20240307-v1:0`
  - `claude-3-opus-20240229-v1:0`
  - `claude-3-5-sonnet-20240620-v1:0`
- **Meta Llama 3**:
  - `llama3-1-405b-instruct-v1:0`
  - `llama3-1-70b-instruct-v1:0`
  - `llama3-1-8b-instruct-v1:0`
  - `llama3-local` (local Llama model based on `llms_ft.py` configurations)
  - `llama3-local-ft` (fine-tuned local Llama model)
  

### Running Evaluation

Example experiments:

```sh
..\venv\Scripts\python src\ie.py --dataset test_small --model anthropic.claude-3-haiku-20240307-v1:0 --ie_prompt ie-v3-claude
..\venv\Scripts\python src\ie.py --dataset test_small --model meta.llama3-1-70b-instruct-v1:0 --ie_prompt ie-v2-llama
..\venv\Scripts\python src\ie.py --dataset test_small --model human
..\venv\Scripts\python src\ie.py --dataset test_small --model rules
```

For more details on parameters:

```sh
..\venv\Scripts\python src\ie.py --help
```

Results are saved in `./results`.

## Performance
<img src="images/performance_overall.png" alt="Caption for the image" width="800">

*Performance comparison of LLMs, human annotators and rule-based baseline.*

<img src="images/performance_ft.png" alt="Caption for the image" width="800">

*Performance of the fine-tuned Llama 3.1 quantized models.*

<img src="images/prompts_performance.png" alt="Caption for the image" width="500">

*Performance improvement in prompt engeneering evaluated on Claude 3 Haiku.*

<img src="images/prompt_over_time.png" alt="Caption for the image" width="500">

*Prompt performance across three Claude models, highlighting improvements
over time.*

<img src="images/summary_plot.png" alt="Caption for the image" width="600">

*Summary plot of the tested models excluding the fine-tuned ones.*