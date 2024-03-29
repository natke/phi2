import argparse
import time
import onnxruntime_genai as og

argparser = argparse.ArgumentParser()
argparser.add_argument('--name', type=str, default='microsoft/phi-2/int4-cpu', help='Phi-2 model to run')
argparser.add_argument('--device', type=str, default='cpu', help='cpu, cuda etc')

name = argparser.parse_args().name
device = argparser.parse_args().device

print(f"Loading model... {name}")
model=og.Model(f'../models/{name}')
print("Model loaded")

tokenizer = og.Tokenizer(model)

prompt = '''Instruct: Summarize the following in a few sentences: A new study has found that evolution is not as unpredictable as previously thought, which could allow scientists to explore which genes could be useful to tackle real-world issues such as antibiotic resistance, disease, and climate change. The study, which is published in the Proceedings of the National Academy of Sciences (PNAS), challenges the long-standing belief about the unpredictability of evolution and has found that the evolutionary trajectory of a genome may be influenced by its evolutionary history, rather than determined by numerous factors and historical accidents. The study was led by Professor James McInerney and Dr. Alan Beavan from the School of Life Sciences at the University of Nottingham, and Dr. Maria Rosa Domingo-Sananes from Nottingham Trent University. \"The implications of this research are nothing short of revolutionary,\" said Professor McInerney, the lead author of the study. \"By demonstrating that evolution is not as random as we once thought, we've opened the door to an array of possibilities in synthetic biology, medicine, and environmental science.\" The team carried out an analysis of the pangenome-the complete set of genes within a given species, to answer a critical question of whether evolution is predictable or whether the evolutionary paths of genomes are dependent on their history and so not predictable today. Using a machine learning approach known as Random Forest, along with a dataset of 2,500 complete genomes from a single bacterial species, the team carried out several hundred thousand hours of computer processing to address the question. After feeding the data into their high-performance computer, the team first made \"gene families\" from each of the gene of each genome. \"In this way, we could compare like-with-like across the genomes,\" said Dr. Domingo-Sananes. Once the families had been identified, the team analyzed the pattern of how these families were present in some genomes and absent in others. \"We found that some gene families never turned up in a genome when a particular other gene family was already there, and on other occasions, some genes were very much dependent on a different gene family being present.\" In effect, the researchers discovered an invisible ecosystem where genes can cooperate or can be in conflict with one another. \"These interactions between genes make aspects of evolution somewhat predictable and furthermore, we now have a tool that allows us to make those predictions,\" adds Dr. Domingo-Sananes. Dr. Beavan said, \"From this work, we can begin to explore which genes 'support' an antibiotic resistance gene, for example. Therefore, if we are trying to eliminate antibiotic resistance, we can target not just the focal gene, but we can also target its supporting genes.\" \"We can use this approach to synthesize new kinds of genetic constructs that could be used to develop new drugs or vaccines. Knowing what we now know has opened the door to a whole host of other discoveries.\" The implications of the research are far-reaching and could lead to: * Novel Genome Design-allowing scientists to design synthetic genomes and providing a roadmap for the predictable manipulation of genetic material. * Combating Antibiotic Resistance-Understanding the dependencies between genes can help identify the 'supporting cast' of genes that make antibiotic resistance possible, paving the way for targeted treatments. * Climate Change Mitigation-Insights from the study could inform the design of microorganisms engineered to capture carbon or degrade pollutants, thereby contributing to efforts to combat climate change. * Medical Applications-The predictability of gene interactions could revolutionize personalized medicine by providing new metrics for disease risk and treatment efficacy.\nOutput:
'''

tokens = tokenizer.encode(prompt)

print(tokens)

params=og.GeneratorParams(model)
params.set_search_options({"max_length":200})
params.input_ids = tokens

start_time=time.time()
output_tokens=model.generate(params)
run_time=time.time()-start_time

print(f"Tokens: {len(output_tokens)} Time: {run_time:.2f} Tokens per second: {len(output_tokens)/run_time:.2f}")

text = tokenizer.decode(output_tokens)

print("Output:")
print(text)

