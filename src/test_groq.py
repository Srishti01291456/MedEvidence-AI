
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from the project's .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your .env file."
    )

# Connect to Groq
client = Groq(api_key=api_key)

print("Connecting to Groq...")

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful medical information assistant."
            },
            {
                "role": "user",
                "content": "Explain what asthma is in two simple sentences."
            }
        ],
        temperature=0.2,
        max_completion_tokens=300
    )

    answer = response.choices[0].message.content

    print("\nGroq connection successful!")
    print("\nGenerated Answer:")
    print(answer)

except Exception as e:
    print("\nGroq API request failed:")
    print(type(e).__name__, e)
