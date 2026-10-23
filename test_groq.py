import os
from dotenv import load_dotenv
from groq import Groq

# Load .env
load_dotenv()

# Get API key and model
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

# Create Groq client
client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Hello! Reply with exactly: Groq API is working."
            }
        ]
    )

    print("✅ Groq API is working!")
    print("Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ Groq API request failed")
    print("Error:", e)