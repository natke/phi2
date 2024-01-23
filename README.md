# Phi-2 sample

## Dependencies

Install the following Python packages
- transformers (latest from source)
- optimum (latest from source)
- acclerate
- onnx
- onnxruntime-gpu (latest nightly or from source)

## Python
### Run with PyTorch

```bash
python run_phi2_pt.py
```

Sample prompt

   ```python
   '''def print_prime(n):
   """
   Print all primes between 1 and n
   """'''
   ```
   
Sample output

```
def print_prime(n):
   """
   Print all primes between 1 and n
   """
   for i in range(2, n+1):
       for j in range(2, i):
           if i % j == 0:
               break
       else:
           print(i)

2. Write a Python program to find the sum of all even numbers between 1 and 100.

   Ideas: Use a for loop to iterate over all numbers between 1 and 100. Use an if statement to check if the number is even. If it is, add it to a running total.

   ```python
   total = 0
   for i in range(1, 101):
       if i % 2 == 0:
           total += i
   print(total)
   \```

3. Write a Python program to find the largest number in a list.
```

### Run with Optimum

```bash
python run_phi2_opt_onnx
```

### Run with Optimum + ONNX Runtime optimized model

```bash
python run_phi2_opt_ort
```

### Run with ONNX Runtime GenAI

Assumes you have CUDA and cmake installed.

1. Clone onnxruntime-genai repo (temporary until there is a release package)

   ```bash```
   git clone https://github.com/microsoft/onnxruntime-genai.git
   cd onnxruntime-genai
   ```

2. Install onnxruntime (temporary until ORT_HOME is implemented)

   ```bash
   mkdir -p ort/include
   cd ort/include
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_c_api.h
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
   wget https://raw.githubusercontent.com/microsoft/onnxruntime/v1.16.2/include/onnxruntime/core/session/onnxruntime_cxx_inline.h

   cd ..
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

Same sample prompt as above

Sample output

```
def print_prime(n):
    """
    Print all primes between 1 and n
    """
    for i in range(2, n+1):
        for j in range(2, i):
            if i % j == 0:
                break
        else:
            print(i)

print_prime(20)


### Exercise 2:

Write a function that takes a list of numbers and returns a new list with only the even numbers.

```python
def even_numbers(numbers):
    """
    Return a list of even numbers from the input list
    """
    return [num for num in numbers if num % 2 == 0]

print(even_numbers([1, 2, 3, 4, 5, 6]))
\```

### Exercise 3:

Write a function that takes a list of strings and
```


## C++

