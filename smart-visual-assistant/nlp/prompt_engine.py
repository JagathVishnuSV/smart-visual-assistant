import google.generativeai as genai

# Initialize Gemini API
genai.configure(api_key="AIzaSyAMR_qFvjtYZ-SRmHW3x8VfbuGMF0m8JWc")

model = genai.GenerativeModel("gemini-2.5-flash")

def build_prompt(query,objects):
    object_str = ", ".join(objects)
    return f"User asked: '{query}'.The camera sees: {object_str}. How should the assistant respond?"

def dummy_llm_response(prompt):
    if not prompt.strip():
        return "Prompt is empty."

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {e}"