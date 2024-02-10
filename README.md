# Phi-2 sample

## Dependencies

Install the following Python packages
- transformers (latest from source)
- optimum (latest from source)
- accelerate
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

Assumes you have cmake installed.

1. Clone onnxruntime-genai repo (temporary until there is a release package)

   ```bash```
   git clone https://github.com/microsoft/onnxruntime-genai.git
   cd onnxruntime-genai
   ```

2. Install onnxruntime (temporary until ORT_HOME is implemented)

   ```bash
   mkdir -o ort
   cd ort
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-gpu-1.17.0.tgz
   tar xvzf onnxruntime-linux-x64-gpu-1.17.0.tgz
   mkdir -p include
   mkdir -p lib
   cp onnxruntime-linux-x64-gpu-1.17.0/include/onnxruntime_c_api.h include
   cp onnxruntime-linux-x64-gpu-1.17.0/lib/libonnxruntime*.so* lib
   ```

   Or copy equivalent files from a local source build

   ```
   ./build.sh --use_cuda --build_shared_lib --build_wheel --skip_tests --parallel
   ```

3. Build onnxruntime-genai

   Change back into root directory of onnxruntime-genai

   ```bash
   cd ..
   python build.py
   ```

4. Install Python package

   ```bash
   cd build/wheel
   pip install *.whl
   ```

4. Export the model for the desired precision and execution target

   ```
   
   ```

5. Run the script to generate text

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

Assumes onnxruntime-genai build steps

1. 

