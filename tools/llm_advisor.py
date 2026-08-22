import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def generate_advice(disease, severity):
    prompt = f"""
    A plant has {disease} with {severity} severity.

    Explain:
    - What the disease is
    - How to treat it (2-3 steps)
    - How to prevent it

    Keep it simple and clear.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    
    return response.text

if __name__ == "__main__":
    result = generate_advice("Powdery Mildew", "Moderate")
    print(result)
