import os
import core_functions
import json
import config
import google.generativeai as genai
import logging
import variables


def create_gemini_model():
    """Create and configure Gemini model with instructions and tools"""
    try:
        # Configure Gemini with API key
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        
        # Create model with instructions
        model = genai.GenerativeModel(variables.GEMINI_MODEL_NAME)
        
        # Set up the system prompt with instructions and examples
        system_prompt = f"{config.assistant_instructions}\n\n### Examples:\n{config.examples}"
        
        print("✅ Gemini model configured successfully")
        return model, system_prompt
        
    except Exception as e:
        print(f"❌ Error configuring Gemini model: {e}")
        return None, None

def get_model_with_tools():
    """Get Gemini model configured with function calling tools"""
    model, system_prompt = create_gemini_model()
    
    if model:
        # For now, return the basic model without tools
        # Function calling will be handled manually in the response processing
        logging.info("✅ Gemini model configured successfully")
        return model
    
    return None
