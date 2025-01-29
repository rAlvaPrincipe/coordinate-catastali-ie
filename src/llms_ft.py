# https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing#scrollTo=kR3gIAX-SM2q
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from dataset import Dataset
from unsloth import FastLanguageModel

class FT_Llama:

    def __init__(self):
        self.max_seq_length = 11200
        self.dtype = None
        self.load_in_4bit = True

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", #"unsloth/llama-3-8b-Instruct-bnb-4bit", #"unsloth/llama-3-8b-Instruct-bnb-4bit", #"unsloth/llama-3-8b-bnb-4bit",
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        self.EOS_TOKEN = self.tokenizer.eos_token
        self.alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def train(self, dataset):
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = 60,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
            ),
        )
        trainer.train()

    def build_prompt(self, instruction, input, output, add_eos_tok=False):
        prompt = self.alpaca_prompt.format(instruction, input, output)
        if add_eos_tok:
            prompt += self.EOS_TOKEN
        return prompt


    def ask(self, prompt):
       # with open("src/json_custom.ebnf", "r") as file:
       #     grammar_str = file.read()
       # grammar = IncrementalGrammarConstraint(grammar_str, "root", self.tokenizer)
       # grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

        FastLanguageModel.for_inference(self.model)
        inputs = self.tokenizer([prompt], return_tensors = "pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens = 2048, use_cache = True)
        #outputs = self.model.generate(**inputs, max_new_tokens = 2048, use_cache = True, logits_processor=[grammar_processor])

        response = self.tokenizer.batch_decode(outputs)
        response = response[0]
        response = response[response.find("### Response:"): -1]
        return response



    def formatting_prompts_func(self, examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = self.build_prompt(instruction, input, output, True)
            texts.append(text)
        return {"text": texts, }

    def adapt_data_to_llama(self, data):
        return data.map(ftl.formatting_prompts_func, batched = True,)


    def save(self):
        self.model.save_pretrained("lora_model") # Local saving
        self.tokenizer.save_pretrained("lora_model")


    def load_ft_model(self):
        model_name = "llms__lora_model_60epochs_4096toks"
        print("LOADING FT LLM: " + model_name)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )


if __name__ == "__main__":
    ftl = FT_Llama()
    dataset = Dataset()
    data = ftl.adapt_data_to_llama(dataset.get_ft_dataset())
    ftl.train(data)
    ftl.save()
