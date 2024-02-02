import onnx
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='model.onnx', help='Path to ONNX model file')

model = argparser.parse_args().model

model = onnx.load(model)
input_names = [i.name for i in model.graph.input]
output_names = [o.name for o in model.graph.output]
print("Inputs: ", input_names)
print("Outputs: ", output_names)