import datetime
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

name = "microsoft/phi-2"
prompt = '''def print_prime(n):
    """
    Print all primes between 1 and n
    """'''

device = "cuda"

if device == "cuda":
    provider = "CUDAExecutionProvider"
else:
    provider = "CPUExecutionProvider"
    

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", use_auth_token=True)

print("Loading model ...")
model = ORTModelForCausalLM.from_pretrained(
       f'models/{name}',
       file_name=f"model.onnx",
       use_auth_token=True,
       provider=provider,
       use_cache=False,
       use_io_binding=False
    )

print("Running tokenizer ...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

num_prompt_tokens = inputs.input_ids.shape[1]

print("Running generate ...")
# Generate
start_time = datetime.datetime.now()  
generate_ids = model.generate(**inputs, max_length=200)   
num_tokens = generate_ids.size(dim=1)
num_new_tokens = num_tokens - num_prompt_tokens
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0] 
end_time = datetime.datetime.now()

print(output)  
print(len(output.split(" ")))
