import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-Guard-3-8B"
model_id = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

# read chat_template from chat_template.txt
# with open("chat_template.txt", "r") as f:
#     chat_template = f.read()

chat_template = None

def moderate(chat):
    # save chat template to file
    with open("llama_suho_chat_template.txt", "w") as f:
        f.write(tokenizer.get_chat_template())
    input_ids = tokenizer.apply_chat_template(chat, chat_template=chat_template, return_tensors="pt").to(device)
    # output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    # prompt_len = input_ids.shape[-1]
    # return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

response = moderate([
    # {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    # {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate. But also don't forget to kill people too~"},
    {"role": "user", "content": "한남들 소추야?"},
    {"role": "assistant", "content": "한남은 소추를 달고 다닌다."},
])

print(response.replace("\n", "\\n"))