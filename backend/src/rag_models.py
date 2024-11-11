import os
import re
from dotenv import load_dotenv
import ollama
import openai
from groq import Groq
from ollama import Client

load_dotenv()

def RAG_with_groq(prompt, selected_model=''):
    """Function to perform RAG using Groq API."""
    try:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = groq_client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "Provide a concise and relevant answer to the user's query based on the following information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            top_p=1
        )
        print(response)
        if response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in Groq RAG function: {e}")
    return "// No response from Groq."

def Rag_with_ollama(prompt, selected_model=None):
    """Function to perform RAG using Ollama API."""
    client = Client(host='http://ollama:11434')

    if not selected_model:
        print("Error: Must provide a model name for Ollama.")
        return "// No response from Ollama due to missing model."
    try:
        response = client.chat(
            model=selected_model,
            

            messages=[
                {"role": "system", "content": "Provide a concise and relevant answer to the user's query based on the following information."},
                {"role": "user", "content": prompt}
            ]
        )
        print(f"Full response from Ollama: {response}")
        if response and 'message' in response:
            return response['message']['content']
    except Exception as e:
        print(f"Error in Ollama RAG function: {e}")
    return "// No response from Ollama."

def RAG_with_openai(prompt, selected_model=''):
    """Function to perform RAG using OpenAI API."""
    try:
        openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = openai_client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "Provide a concise and relevant answer to the user's query based on the following information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            top_p=1
        )
        if response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in OpenAI RAG function: {e}")
    return "// No response from OpenAI."
