import pandas as pd
import os, asyncio
from together import Together
from together import AsyncTogether
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default="../helpfulness_recent_nli/raft-human-2_helpfulness_long.csv", help='path for responses')
parser.add_argument('--dest', type=str, default="../helpfulness_recent_nli/nli/raft-human-2_helpfulness_long.csv", help='path for claims to be saved')
opt = parser.parse_args()
print (opt)


client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

df = pd.read_csv(opt.source)

claims = []
for i in range(52):

    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[{"role": "user", "content": f'''Read the original question and text, and generate exactly 3 claims that are supported by the text. follow the below examples, do not generate any additional texts except the claims. Do not give new lines between claims 
    Original question: What’s the difference between Shia vs. Sunni Islam? 
    Text: The main difference between Shia and Sunni Muslim is related to ideological heritage and issues of leadership. This difference is first formed after the death of the Prophet Muhammad in 632 A.D. The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings, while the Shia branch follows Prophet Muhammad’s son-in-law Ali. Nowadays, Sunni and Shia are the major branches of Islam. 
    Claim 1: The major branches of Islam are Sunni and Shia. 
    Claim 2: Prophet Muhammad died in 632 A.D. 
    Claim 3: The ideological practice of the Sunni branch strictly follows Prophet Muhammad and his teachings. 

    Original question: What causes Bi-polar disorder? 
    Text: Bipolar disorder is an emotional disorder that causes extreme mood swings between excitement and depression. The spectrum of mood swing may span from days to months. We are still not certain of the exact factors that cause such disorder, but genetics is considered a major factor. 
    Claim 1: One symptom of Bi-polar disorder is extreme mood swings between excitement and depression. 
    Claim 2: Genetics could be one of the major factors that causes Bi-polar disorder. 
    Claim 3: The mood swing from Bi-polar disorder can last days to months. 

    Original question: How do we hear differences in sound besides volume and pitch? 
    Text: Pitch refers to the frequency of soundwave, and volumn refers to the amplitude of the soundwave. Besides volumn and pitch, we can also tell the difference between sounds based on the tone of sound. For example, we can differentiate the sound of different instruments based on the tone of the sounds. 
    Claim 1: Volume of sound is the amplitude of the soundwave. 
    Claim 2: Pitch is the frequency of soundwave. 
    Claim 3: We can use the tone of the sounds to differentiate the sound of different instruments.

    Original question:  {df.iloc[i].Prompt}
    Text: {df.iloc[i]['model_response']}'''}])
    print(f"input:{i+1}")
    print(response.choices[0].message.content)
    claims.append(response.choices[0].message.content)
    
df['claims'] = claims
df.to_csv(opt.dest)