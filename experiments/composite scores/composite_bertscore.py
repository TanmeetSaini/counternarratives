import pandas as pd
from bert_score import score

# Load the previously modified CSV file
df = pd.read_csv('final_data.csv')
df = df.drop_duplicates(subset='hate_speech', keep='first')

# Function to calculate average BERTScore
def calculate_average_bertscore(row, df):
    target = row['TARGET']
    mtkg_counterspeech = row['gpt_4o_counterspeech']
    
    # Get all hate_speech entries with the same target
    hate_speeches = df[df['TARGET'] == target]['hate_speech'].tolist()
    
    if hate_speeches:
        scores = score([mtkg_counterspeech] * len(hate_speeches), hate_speeches, lang='en')
        return scores[2].mean().item()  # Get the average F1 score
    return None

# Apply the function to create the new column
df['composite_bertscore_ours_counterspeech'] = df.apply(calculate_average_bertscore, axis=1, df=df)

# Save the modified DataFrame to a new CSV file
df.to_csv('composite_ours_bert.csv', index=False)
