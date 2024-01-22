from transformers import AutoTokenizer
import onnx
from onnxruntime_extensions import OrtPyFunction, gen_processing_models

# phi-2 tokenizer
phi2_hf_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", use_fast=False)
phi2_tokenizer, phi2_detokenizer = gen_processing_models(phi2_hf_tokenizer, pre_kwargs={}, post_kwargs={})
onnx.save_model(phi2_tokenizer, "models/microsoft/phi-2/tokenizer.onnx")
onnx.save_model(phi2_detokenizer, "models/microsoft/phi-2/detokenizer.onnx")
