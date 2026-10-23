import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = os.getenv("GROQ_MODEL")


def analyze_sentiment(text):

    prompt = f"""
Analyze the sentiment of the following text.

Text:
{text}

Classify it into exactly one of these categories:
Positive
Negative
Neutral

Give your response in this format:

Sentiment: <Positive/Negative/Neutral>
Explanation: <short explanation>
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content