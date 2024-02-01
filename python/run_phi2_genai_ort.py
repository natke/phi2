import argparse
import time
import onnxruntime_genai as og

argparser = argparse.ArgumentParser()
argparser.add_argument('--name', type=str, default='microsoft/phi-2/fp16-cuda', help='Phi-2 model to run')

name = argparser.parse_args().name

#device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA


print("Loading model...")
model=og.Model(f'../models/{name}', device_type)
print("Model loaded")

tokenizer = model.CreateTokenizer()

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokens = tokenizer.encode(prompt)

params=og.SearchParams(model)
params.max_length = 200
params.input_ids = tokens[0]

start_time=time.time()
output_tokens=model.Generate(params)
run_time=time.time()-start_time

print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

text = tokenizer.batch_decode(output_tokens)[0]

print("Output:")
print(text)

