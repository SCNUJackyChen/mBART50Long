import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from longformer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = MBartForConditionalGeneration.from_pretrained("/content/mbart-large-50").to(device)
model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained("/content/mbart-large-50").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
model.eval()

def translate_one(source_text: str, num_beams=1) -> str:
  inputs = tokenizer(source_text, return_tensors="pt", padding="max_length").to(device)
  # print(inputs["input_ids"].shape)
  translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"], num_beams=num_beams)
  res = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
  return res[0]

print(translate_one("Very happy to be here this morning to see the start of the inoculations for COVID-19 for our healthcare workers and frontline workers."))