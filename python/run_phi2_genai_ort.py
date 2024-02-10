import argparse
import time
import onnxruntime_genai as og

argparser = argparse.ArgumentParser()
argparser.add_argument('--name', type=str, default='microsoft/phi-2/int4-cuda', help='Phi-2 model to run')
argparser.add_argument('--device', type=str, default='cpu', help='cpu, cuda etc')

name = argparser.parse_args().name
device = argparser.parse_args().device

if (device == 'cuda'):
  device_type = og.DeviceType.CUDA
else:
  device_type = og.DeviceType.CPU

print(f"Loading model... {name} on {device_type}")
model=og.Model(f'../models/{name}', device_type)
print("Model loaded")

tokenizer = model.create_tokenizer()

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

print(tokens)

params=og.search_params(model)
params.max_length = 200
params.input_ids = tokens

start_time=time.time()
output_tokens=model.generate(params)
run_time=time.time()-start_time

print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

text = tokenizer.decode(output_tokens)

print("Output:")
print(text)

