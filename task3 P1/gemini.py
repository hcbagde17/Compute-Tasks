import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
while True:
    try:
        prompt = input("Enter your prompt: ")
        if prompt=='q':
            break
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        print(response.text)
    except Exception as e:
        print(f"An error occurred: {e}")