import onnxruntime_genai as og

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
