import argparse
import numpy as np
import time
import onnxruntime
import onnxruntime_extensions as extensions
import onnxruntime_genai as og

argparser = argparse.ArgumentParser()
argparser.add_argument('--name', type=str, default='microsoft/phi-2/int4', help='Phi-2 model to run')

name = argparser.parse_args().name

model=og.Model(f'models/microsoft/phi-2/int4', og.DeviceType.CUDA)

prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

tokenizer = model.CreateTokenizer()

tokens = tokenizer.Encode(prompt)

params=og.SearchParams(model)
params.input_ids = tokens[0]

output = model.Generate(params)

print(tokenizer.Decode(output))

print("Output:")
print(output[0][0])
