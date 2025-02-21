import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file (if needed for other configurations)
load_dotenv()

# LangSmith Tracing (Optional, remove if not needed)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Simple Q&A CHATBOT"

# Load GPT-2 model for text generation (no context required)
chatbot = pipeline("text-generation", model="gpt2")

# Define LangChain Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's question concisely."),
    ("user", "{question}"),
])

def generate_response(question):
    # Format the prompt with the user's question
    formatted_prompt = prompt.format(question=question)
    
    # Generate response based on the formatted prompt
    response = chatbot(formatted_prompt, max_length=50, num_return_sequences=1)
    
    # Return the generated text
    return response[0]['generated_text']

# Example usage
question = "What is the capital of France?"
answer = generate_response(question)
print(answer)
