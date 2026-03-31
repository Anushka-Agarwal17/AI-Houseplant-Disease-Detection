from google import genai

client = genai.Client(api_key="AIzaSyAsKuuZQ8L9FPFSPlC4PWpFA1a3kRCkepA")

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
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return response.text

if __name__ == "__main__":
    result = generate_advice("Powdery Mildew", "Moderate")
    print(result)
