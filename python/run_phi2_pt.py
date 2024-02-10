import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, trust_remote_code=True)
model.save_pretrained("saved_model")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.save_pretrained("saved_model")

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
print(outputs.shape)
text = tokenizer.batch_decode(outputs)[0]
print(text)