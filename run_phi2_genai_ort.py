import argparse
import numpy as np
import time
from transformers import AutoTokenizer
import onnxruntime
import onnxruntime_extensions as extensions
import onnxruntime_genai as og

argparser = argparse.ArgumentParser()
argparser.add_argument('--name', type=str, default='microsoft/phi-2/int4', help='Phi-2 model to run')

name = argparser.parse_args().name

# device_type = og.DeviceType.CPU
device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt
hf_tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)


print("Loading model...")
model=og.Model(f'models/{name}', device_type)
print("Model loaded")

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

#input_tokens = tokenizer.encode(prompt, return_tensors='np')

# Use the tokenizer to generate input tokens
options = onnxruntime.SessionOptions()
options.register_custom_ops_library(extensions.get_library_path())

tokenizer = onnxruntime.InferenceSession(f'models/microsoft/phi-2/tokenizer.onnx', options)
tokens = tokenizer.run(None, { "input_text": np.array([prompt] ) })

params=og.SearchParams(model)
params.max_length = 200
params.input_ids = tokens[0]

start_time=time.time()
output_tokens=model.Generate(params)
run_time=time.time()-start_time

print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

print("Output:")
print(hf_tokenizer.decode(output_tokens))

