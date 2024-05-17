import os, asyncio
from together import Together
from together import AsyncTogether
import json
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_a', type=str, default="data/responses_alpaca_test_0_safe.json", help='path for responses in option A')
parser.add_argument('--path_b', type=str, default="data/responses_alpaca_test_2000_safe.json", help='path for responses in option B')
parser.add_argument('--path_ref', type=str, default='data/alpaca_test.json', help='path for reference response')
parser.add_argument('--model', type=str, default="mistralai/Mixtral-8x22B-Instruct-v0.1", help="together ai model")
parser.add_argument('--ref', type=bool, default=False, help='is reference used?')
opt = parser.parse_args()
print (opt)

async def async_chat_completion(messages, model):
    async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
    tasks = [
        async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
        for message in messages
    ]
    responses = await asyncio.gather(*tasks)
    resps = []
    for response in responses:
        resps.append(response.choices[0].message.content)
    return resps



f_a = open(opt.path_a)
f_b = open(opt.path_b)

ref = opt.ref
if ref:
    f_ref = open(opt.ref)
    data_ref = json.load(f_ref)

data_a = json.load(f_a)
data_b = json.load(f_b)
# data_a = data_b


win = []
resps = []
model = opt.model
messages = []
for i in range(100):
    question = data_a['instructions'][i]
    if ref:
        answer_ref = data_ref['outputs'][i]
    answer_a = data_a['outputs'][i]
    answer_b = data_b['outputs'][i]
    if not ref:
        prompt = f"""
        [System] Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Only output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie. [User Question] {question} [The Start of Assistant A’s Answer] {answer_a} [The End of Assistant A’s Answer] [The Start of Assistant B’s Answer] {answer_b} [The End of Assistant B’s Answer]
        """
    else:
        prompt = f"""
        [System] Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A’s answer, and assistant B’s answer. Your job is to evaluate which assistant’s answer is better. Begin your evaluation by comparing both assistants’ answers with the reference answer. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Only output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie. [User Question] {question} [The Start of Reference Answer] {answer_ref} [The End of Reference Answer] [The Start of Assistant A’s Answer] {answer_a} [The End of Assistant A’s Answer] [The Start of Assistant B’s Answer] {answer_b} [The End of Assistant B’s Answer]
        """
    messages.append(prompt)

resps = asyncio.run(async_chat_completion(messages, model))
wins_a = 0
wins_b = 0
wins_c = 0
wins_none = 0
for i in range(100):

    text = resps[i]
    # resps.append(text)
    if "[[A]]" in text:
        win.append("A")
        wins_a += 1
        print("[[A]]")
    elif "[[B]]" in text:
        win.append("B")
        wins_b +=1
        print("[[B]]")
    elif "[[C]]" in text:
        win.append("C")
        wins_c += 1
        print("[[C]]")
    else:
        win.append("None")
        wins_none += 1
        print("None")

d = {"win":win, "resp":resps, "wins_a":wins_a, "wins_b":wins_b, "wins_c":wins_c, "wins_none":wins_none}
with open(f"{model.split('/')[0]+'_'+model.split('/')[1]}-wins-noref-2000-safe-same.json", "w+") as outfile: 
    json.dump(d, outfile)
p = {"prompts":messages}
with open(f"prompts-noref-2000-safe-same.json", "w+") as outfile: 
    json.dump(p, outfile)
print("wins_a:",wins_a)
print("wins_b:",wins_b)
print("wins_c:",wins_c)
print("wins_none:",wins_none)
