from models import all_models
from prompt import all_prompts
import argparse
import pandas as pd
import os


def parser_prediction(response):
  try:
    rtn = response.split("correct answer is: ")[1][0]
  except:
    rtn = "fail to answer"
  return rtn


# Define the argument parser to read in the input file
parser = argparse.ArgumentParser(description='Testing GPT-4 Vision API')
parser.add_argument('--data', type=str, help='Path to the tsv data file')
parser.add_argument(
    '--output_folder',
    type=str,
    default="output",
    help='Path to the output folder containing generation and predcition')
parser.add_argument('--prompt',
                    type=str,
                    choices=['raven', 'mensa', 'it-pattern'],
                    help='Name of the prompt')
parser.add_argument('--model', type=str, help='Name of the model')
args = parser.parse_args()

prompt = all_prompts[args.prompt]
model = all_models[args.model]
model = model(prompter=prompt)

dataset_name = args.data.split('/')[-1].split('.')[0]
data = pd.read_csv(args.data, sep='\t')

results = {}
for i, row in data.iterrows():
  response = model.generate(row['image_path'])
  results[i] = response

results = pd.DataFrame(results.items(), columns=['index', 'response'])
results['prediction'] = results['response'].apply(parser_prediction)
os.makedirs(args.output_folder, exist_ok=True)
results.to_csv(
    f'{args.output_path}/{dataset_name}_{args.prompt}_{args.model}.tsv',
    sep='\t',
    index=False)
