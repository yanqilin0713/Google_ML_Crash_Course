# Set up programming environment to use code to send prompts to OpenAI's cloud-hosted service.
from openai import OpenAI
import os

def llm_response(prompt):
    response = client.responses.create(
        model='gpt-4.1-mini',
        input=prompt,
        temperature=0
    )
    text = response.output[0].content[0].text
    usage = response.usage
    return text, usage

# Create a list of reviews.
all_reviews = [
    'The mochi is bad!',
    'Best soup dumplings I have ever eaten.',
    'Not worth the 3 month wait for a reservation.',
    'The colorful tablecloths made me cry!',
    'The pasta was cold.'
]

# Classify the reviews as positive or negative.
all_sentiments = []
for review in all_reviews:
    prompt = f'''
        Classify the following review
        as having either a positive or
        negative sentiment. State your answer
        as a single word, either "positive" or
        "negative":

        {review}
        '''
    text, usage = llm_response(prompt)
    print(text)
    print("Usage:", usage)
    all_sentiments.append(text)

# Count the number of positive and negative reviews
num_positive = 0
num_negative = 0
for sentiment in all_sentiments:
    if sentiment == 'positive':
        num_positive += 1
    elif sentiment == 'negative':
        num_negative += 1
print(f"There are {num_positive} positive and {num_negative} negative reviews.")
