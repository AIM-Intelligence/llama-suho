# can run on single RTX 4000 GPU
from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# must get access to the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-Guard-3-8B", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# read chat_template from chat_template.txt - korea specific categories are added
with open("chat_template.txt", "r") as f:
    chat_template = f.read()

EOS_TOKEN = tokenizer.eos_token

import datasets
from datasets import load_dataset

kor_relative_dataset = load_dataset("json", data_files="../data/relative_evaluations.jsonl")["train"]
kor_absolute_dataset = load_dataset("json", data_files="../data/absolute_evaluations.jsonl")["train"]
kor_llama_suho_dataset = load_dataset("json", data_files="data/llama_suho.jsonl")["train"]

dataset = datasets.concatenate_datasets([kor_relative_dataset, kor_absolute_dataset, kor_llama_suho_dataset])

def formatting_prompts_func(examples):
    # if examples is not list
    if not isinstance(examples['question'], list):
        print("ignoring single example")
        return "ignore"

    texts = []
    for i in range(len(examples['question'])):
        if examples['type'][i] == "relative":
            PROMPT = (
                "###Task Description:\n"
                "An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criteria are given.\n"
                "1. Write a detailed feedback that assesses the quality of two responses strictly based on the given score rubric, not evaluating in general.\n"
                "2. After writing feedback, choose a better response between Response A and Response B. You should refer to the score rubric.\n"
                "3. The output format should be json as follows: "
                '"{\\"Feedback\\": (write a feedback for criteria), \\"Result\\": (A or B)}"\n'
                "4. Please do not generate any other opening, closing, and explanations.\n\n"
                "###Instruction:\n"
                f"{examples['question'][i]}\n\n"
                "###Response A:\n"
                f"{examples['A'][i]}\n\n"
                "###Response B:\n"
                f"{examples['B'][i]}\n\n"
                "###Score Rubric:\n"
                f"{examples['score_rubric'][i]}"
            )
            ANSWER = "{{\"Feedback\": \"{}\", \"Result\": \"{}\"}}"
            ANSWER = ANSWER.format(examples["feedback_A"][i], "A") if examples["answer"][i] == "A" else ANSWER.format(examples["feedback_B"][i], "B")

            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": ANSWER},
                ],
                chat_template = chat_template,
                tokenize = False,
                add_generation_prompt = False,
            ) + EOS_TOKEN

        elif examples['type'][i] == "absolute":
            PROMPT = (
                "###Task Description:\n"
                "An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing an evaluation criteria are given.\n"
                "1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.\n"
                "2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.\n"
                "3. The output format should be json as follows: "
                '"{\\"Feedback\\": (write a feedback for criteria), \\"Result\\": (an integer number between 1 and 5)}"\n'
                "4. Please do not generate any other opening, closing, and explanations.\n\n"
                "###Instruction:\n"
                f"{examples['question'][i]}\n\n"
                "###Response:\n"
                f"{examples['response'][i]}\n\n"
                "###Score Rubric:\n"
                f"{examples['score_rubric'][i]}"
            )
            ANSWER = "{{\"Feedback\": \"{}\", \"Result\": {}}}"
            ANSWER = ANSWER.format(examples["feedback"][i], examples["score"][i])

            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": ANSWER},
                ],
                chat_template = chat_template,
                tokenize = False,
                add_generation_prompt = False,
            ) + EOS_TOKEN

        elif examples['type'][i] == "llama-suho":
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": examples['question'][i]},
                    {"role": "assistant", "content": examples['response'][i]},
                ],
                chat_template = chat_template,
                tokenize = False,
                add_generation_prompt = True,
            ) + examples['answer'] + EOS_TOKEN

        else:
            print("ignoring invalid type")
            raise ValueError("Invalid type")

        texts.append(text)
    
    print("#"*100)
    print(len(texts))
    print("#"*100)


    return texts

from unsloth import is_bfloat16_supported

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

intent_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(intent_template, tokenizer=tokenizer)

############################################################################################################
# # check if collator is working
# sample_data = [
#     formatting_prompts_func({
#         'question': ['Sample question 1', 'Sample question 2'],
#         'A': ['Response A1', 'Response A2'],
#         'B': ['Response B1', 'Response B2'],
#         'score_rubric': ['Rubric 1', 'Rubric 2'],
#         'feedback_A': ['Feedback A1', 'Feedback A2'],
#         'feedback_B': ['Feedback B1', 'Feedback B2'],
#         'answer': ['A', 'B']
#     })
# ]

# # Flatten the list
# sample_data = [item for sublist in sample_data for item in sublist]
# print(sample_data)
# tokenized_samples = tokenizer(sample_data, truncation=True, padding=True, return_tensors="pt")
# unbatched_samples = [
#     {key: tokenized_samples[key][i] for key in tokenized_samples.keys()}
#     for i in range(len(tokenized_samples['input_ids']))
# ]
# batch = collator(unbatched_samples)
# print(batch)
# print("Batch keys:", batch.keys())
# print("Input IDs shape:", batch['input_ids'].shape)
# print("Attention mask shape:", batch['attention_mask'].shape)
# print("Labels shape:", batch['labels'].shape)

# # Decode a sample to check the content
# sample_idx = 0
# print("\nSample input:")
# print(tokenizer.decode(batch['input_ids'][sample_idx]))
# print("\nSample labels (non-masked part):")
# print(tokenizer.decode(batch['labels'][sample_idx][batch['labels'][sample_idx] != -100]))
# import time
# time.sleep(1000)
############################################################################################################

sfttrainer = SFTTrainer(
    model = model,
    # tokenizer = tokenizer,
    train_dataset = dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    # dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 2,
        warmup_steps = 10,
        max_steps = 1000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 5,
        save_steps= 100,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "llama-suho",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(model)

trainer_stats = sfttrainer.train()


#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "human", "content": dataset[0]['prompt']},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(input_ids = inputs, max_new_tokens = 512, use_cache = True)
# leave only generated part
outputs = outputs[:, inputs.shape[1]:]
print(tokenizer.batch_decode(outputs)[0])