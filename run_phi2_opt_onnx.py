import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

torch.set_default_device("cuda")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

model = ORTModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True, export=True)

model.save_pretrained(f"models/microsoft/phi-2")

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
print(outputs.shape)
text = tokenizer.batch_decode(outputs)[0]
print(text)
