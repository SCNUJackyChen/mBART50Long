# Mbart-large-50 (Longformer version)

A [longformer](https://github.com/allenai/longformer) modification of mbart model from Huggingface checkpoint [facebook/mbart-large-50-many-to-many-mmt](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)

## How to use

### 0. Install
```
pip install transformers
pip install sentencepiece
```
### 1. convert mbart to its long version
```
python convert_bart_to_longformerencoderdecoder.py \
--save_model_to path-to-save-new-model \
--base_model "facebook/mbart-large-50-one-to-many-mmt" \
--tokenizer_name_or_path "facebook/mbart-large-50-one-to-many-mmt" 
```

### 2. model loading & inference
```python
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from longformer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained(path-to-save-new-model).to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
model.eval()

def translate_one(source_text: str, num_beams=1) -> str:
  inputs = tokenizer(source_text, return_tensors="pt", padding="max_length").to(device)
  # print(inputs["input_ids"].shape)
  translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"], num_beams=num_beams)
  res = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
  return res[0]

print(translate_one("hello world", 2))
```
For more details, [![this notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19yTi7b4KUvMNl8msRSr9a5p2CZa0V6ES?usp=sharing)
