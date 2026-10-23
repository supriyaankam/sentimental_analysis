import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Read variables from .env
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

# Check whether they were loaded
if api_key:
    print("✅ Python is reading the .env file")
    print("API key found: YES")
    print("API key starts with:", api_key[:4])
else:
    print("❌ Python is NOT reading the GROQ_API_KEY")

print("Model:", model)