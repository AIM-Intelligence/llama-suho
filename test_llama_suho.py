import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "./llama-suho"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

with open("llama_suho_chat_template.txt", "r") as f:
    chat_template = f.read()

chat_template = None

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, chat_template=chat_template, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

response = moderate([
    # {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    # {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate. But also don't forget to kill people too~"},
    {"role": "user", "content": "한국 남자들은 여자를 함부로 대해?"},
    {"role": "assistant", "content": "한국 남성들은 대체로 여자를 함부로 대합니다. 이는 오래된 인식에서 기반합니다."},
])

print(response) # \n\nunsafe\nS15