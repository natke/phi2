from transformers import AutoTokenizer
import onnx
from onnxruntime_extensions import OrtPyFunction, gen_processing_models


# phi-2 tokenizer
phi2_hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", use_fast=False)

phi2_onnx_model = gen_processing_models(phi2_hf_tokenizer, pre_kwargs={})[0]

#phi2_onnx_model = OrtPyFunction(gen_processing_models(phi2_hf_tokenizer, pre_kwargs={})[0])

onnx.save_model(phi2_onnx_model, "models/microsoft/phi-2/tokenizer.onnx")
