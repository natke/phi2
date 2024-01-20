# Phi-2 sample

## Dependencies

Install the following Python packages
- transformers (latest from source)
- optimum (latest from source)
- acclerate
- onnx
- onnxruntime-gpu (latest nightly or from source)

## Run with PyTorch

```bash
python run_phi2_pt.py
```

## Run with Optimum

```bash
python run_phi2_opt_onnx
```

## Run with Optimum + ONNX Runtime optimized model

```bash
python run_phi2_opt_ort
```

## Run with ONNX Runtime GenAI

Assumes you have CUDA and cmake installed.

1. Clone onnxruntime-genai repo (Temporaty until there is a release package)

   ```bash```
   git clone https://github.com/microsoft/onnxruntime-genai.git
   cd onnxruntime-genai
   ```

2. Install onnxruntime (temporary until ORT_HOME is implemented)

   ```bash
   mkdir -p ort
   cd ort
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_c_api.h
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_cxx_inline.h

   wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.2/onnxruntime-linux-x64-gpu-1.16.2.tgz
   tar xvzf onnxruntime-linux-x64-gpu-1.16.2.tgz
   cp onnxruntime-linux-x64-gpu-1.16.2/lib/libonnxruntime*.so* .
   ```

3. Build onnxruntime-genai

   Change back into root directory of onnxruntime-genai

   ```bash
   cd ..
   bash build.sh
   ```

4. Set python path so onnxruntime-genai lib can be found by Python (temporary)

   ```bash
   export PYTHONPATH=`pwd`/build
   ```

5. Run the script to generate text with Llama

   ```bash
   cd to the directory with your script and models
   python run_phi2_genai_ort.py
   ```