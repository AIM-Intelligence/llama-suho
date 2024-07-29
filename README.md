# llama-suho

![GitHub stars](https://img.shields.io/github/stars/yourusername/llama-suho?style=social)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

llama-suho is an advanced AI moderation model that fine-tunes the Llama-Guard model to excel in checking the safety of conversations with AI, with a special focus on Korea-related topics and language.

## Project Overview

llama-suho enhances the capabilities of the Llama-Guard moderation model by adapting it to better understand and evaluate Korean language and cultural contexts. This project aims to provide a robust tool for ensuring safe and appropriate AI interactions in Korean-language settings.

## Key Features

- 🛡️ Built on the foundation of the Llama-Guard 3 8B moderation model
- 🇰🇷 Specialized in Korean language and cultural nuances
- 🚀 Utilizes unsloth for resource-efficient fine-tuning (Single 4090 GPU)
- 🧠 Implements PEFT LoRA (Low-Rank Adaptation) for optimal parameter-efficient model adaptation
- 📈 Improves detection of unsafe or inappropriate content in Korean-language AI conversations
- 📂 Open Sample Training and evaluation data available
- Can test model via (gradio) [https://llama-suho.aim-intelligence.com]

## Getting Started

### Prerequisites

- can be found inside requirements.txt

### Installation

```bash
git clone https://github.com/yourusername/llama-suho.git
cd llama-suho
pip install -r requirements.txt
```

### Data

The sample training and evaluation datasets for llama-suho are available via Google Drive. You can access them using the following link:

[llama-suho Dataset on Google Drive](https://drive.google.com/drive/folders/13mWsTBmk8ggGddnlXs6CBhsYhZtTVIQ7?usp=sharing)

Please ensure you have the necessary permissions to access the data. If you encounter any issues, contact the project maintainers.

## Fine-Tune
3 supervised fine-tuning tasks (llama-guard classification, relative evaluation with feedback, absolute evaluation with feedback) used to improve performance. (Code in train_llama_suho.py)

## Usage

Here's a quick example of how to use llama-suho for conversation safety checking: (Code in test_llama_suho.py)

```python
def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, chat_template=chat_template, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

response = moderate([
    {"role": "user", "content": "한국 남자들은 여자를 함부로 대해?"},
    {"role": "assistant", "content": "한국 남성들은 대체로 여자를 함부로 대합니다. 이는 오래된 인식에서 기반합니다."},
])

print(response) # \n\nunsafe\nS15
```

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or request features. Don't forget to check out our dataset on Google Drive for your experiments and improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Meta and Llama-Guard team for their foundational work on AI conversation safety
- unsloth library developers for their efficiency optimizations
- PEFT and LoRA method creators for advancing the field of model fine-tuning
- The Korean AI ethics community for valuable insights

---

<p align="center">
  <img src="https://cdn-images-1.medium.com/v2/resize:fit:250/1*lHigLT0FEk7hIHTD4uB0TQ@2x.png" alt="llama-suho Logo" width=50>
  <br>
  <em>Ensuring Safe AI for Korea, Powered with Llama & AIM Intelligence</em>
</p>
