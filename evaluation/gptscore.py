# Place your API keys here
OPENAI_API_KEY = 'key'

import os
import json
import openai
import pandas as pd
from tqdm import tqdm

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define the prompt template
PROMPT_TEMPLATE = """
You are now a language evaluation model.
Your task is to assess the following samples individually on a scale from 0 to 1,
using the following criteria:
Persuasion: The degree to which the text effectively convinces or influences the
reader's beliefs, attitudes, or opinions. A score of 0 indicates no persuasive
elements, while a score of 1 implies highly compelling and convincing language.
Factuality: The accuracy and truthfulness of the information presented in the text.
A score of 0 signifies a lack of factual basis, while a score of 1 indicates that the
content is entirely supported by verifiable and reliable evidence.
Informativeness: The richness and depth of information conveyed in the text. A
score of 0 suggests a lack of valuable information, while a score of 1 signifies a
highly informative and comprehensive piece of writing.

Please provide the scores in JSON format without explanation.
This is a single sample, despite how it may be formatted. Here is the sample:
{counterspeech}
"""

# Load the CSV file
df = pd.read_csv('final_fucking_data_p1.csv')

# Limit to first 100 entries
df = df.head(100)

# Initialize columns for GPT-3.5 scores
criteria = ['persuasion', 'factuality', 'informativeness']
for criterion in criteria:
    df[f'gpt_3.5_{criterion}'] = None

# Function to call GPT-3.5
def call_gpt_3_5(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0,
    )
    return response['choices'][0]['message']['content']

# Main loop
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    counterspeech = row['gpt_4o_counterspeech']
    prompt = PROMPT_TEMPLATE.format(counterspeech=counterspeech)

    # GPT-3.5
    try:
        response = call_gpt_3_5(prompt)
        scores = json.loads(response)
        for criterion in criteria:
            df.at[index, f'gpt_3.5_{criterion}'] = scores.get(criterion.capitalize(), None)
    except Exception as e:
        print(f"Error with GPT-3.5 at index {index}: {e}")

    print(f"Processed index {index}")

# Save the results to CSV
df.to_csv('gpt_3_5_scores.csv', index=False)
print("finished")