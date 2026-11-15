import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_llm_client():
    api_key = os.environ.get('LLM_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('LLM_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta/openai/')
    
    if not api_key or api_key in ['your_free_api_key_here', 'your_llm_api_key_here']:
        return None
        
    return OpenAI(api_key=api_key, base_url=base_url)

def get_model_name():
    return os.environ.get('LLM_MODEL', 'gemini-3.6-flash')

def generate_ai_explanation(prediction, confidence, probabilities, model_name):
    client = get_llm_client()
    disclaimer = "\n\n*Disclaimer: This AI-generated explanation is for informational purposes only and does not replace professional medical advice.*"
    
    prompt = f"""
    Analyze the following brain MRI classification result and provide a clear, patient-friendly explanation:
    - Predicted Tumor Category: {prediction}
    - Confidence Score: {confidence}%
    - Model Used: {model_name}
    - Class Probabilities: {probabilities}
    
    Structure your response into:
    1. What this classification means in simple terms.
    2. Interpretation of the confidence score.
    3. Recommended next steps for clinical consultation.
    """
    
    if client:
        try:
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant explaining brain tumor MRI classification results clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=450,
                temperature=0.4
            )
            return response.choices[0].message.content + disclaimer
        except Exception as e:
            print(f"[WARNING] LLM API call failed: {e}")
            return f"AI Analysis Summary:\nThe analysis classified the scan as {prediction} with {confidence}% confidence. (API note: {str(e)}){disclaimer}"
            
    return f"""
**AI Analysis Summary:**
The analysis performed using the **{model_name}** architecture classified the MRI scan as **{prediction}** with **{confidence}%** confidence. Please consult a neurologist for a formal clinical evaluation.
{disclaimer}
"""

def answer_patient_query(question, report_context=None):
    client = get_llm_client()
    
    context_prompt = f"User's Latest MRI Report Context: {report_context}\n\n" if report_context else ""
    full_prompt = context_prompt + question
    
    if client:
        try:
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful, friendly, and unrestricted general-purpose AI assistant integrated into the BrainScan-AI platform. You can answer ANY question the user asks—whether about brain tumors, MRI reports, general health, science, coding, or everyday topics. If the question is medical, gently remind them to consult a doctor for official diagnoses."
                    },
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"API Connection Error: {str(e)}. Please check your API key in the .env file."
            
    return "Free AI API key not configured! Please add your valid Google Gemini API key to your .env file."