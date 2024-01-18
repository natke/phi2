import onnxruntime_genai as og
import numpy as np
import time
from transformers import AutoTokenizer

# device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)

print("Loading model...")
model=og.Model("models/microsoft/phi-2/int4", device_type)
print("Model loaded")

text = input("Input:")

input_tokens = tokenizer.encode(text, return_tensors='np')

params=og.SearchParams(model)
params.max_length = 64
params.input_ids = input_tokens

start_time=time.time()
output_tokens=model.Generate(params)
run_time=time.time()-start_time

print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

print("Output:")
print(tokenizer.decode(output_tokens))

