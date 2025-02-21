import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Set up the GROQ API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ API Key not found in environment variables.")

# Set up the HuggingFace Access Token (For embedding)
hf_access_token = os.getenv("HF_ACCESS_TOKEN")
if not hf_access_token:
    raise ValueError("HuggingFace Access Token not found in environment variables.")

os.environ['HF_ACCESS_TOKEN'] = hf_access_token
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up HuggingFace Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize the ChatGroq LLM with the API Key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template for the LLM
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Chat History: {chat_history}
    
    Question: {input}
    """
)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def create_vector_embedding():
    """Function to create document embeddings and return a FAISS vector store."""
    loader = PyPDFDirectoryLoader("research_papers")
    docs = loader.load()
    if not docs:
        print("No documents found in the 'research_papers' directory.")
        return None

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:50])  # Limit to first 50 docs
    if not final_documents:
        print("No documents were split successfully.")
        return None

    # Create FAISS vector store from the documents and embeddings
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors


def main():
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")
    
    # Create document chain and retriever
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create vectors using the function
    vectors = create_vector_embedding()
    
    # If vectors weren't created, exit
    if not vectors:
        print("Failed to create document vectors.")
        return
    
    # Create the retriever and retrieval chain
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    while True:
        # Get user input for the query
        user_prompt = input("\nYou: ")
        
        if user_prompt.lower() == "exit":
            print("Goodbye!")
            break

        # Invoke the retrieval chain with memory
        response = retrieval_chain.invoke({
            "input": user_prompt,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        })

        bot_reply = response['answer']
        print(f"\nAI: {bot_reply}")

        # Store conversation history in memory
        memory.save_context({"input": user_prompt}, {"output": bot_reply})

        # Document similarity search (display documents similar to the answer)
        print("\nDocument Similarity Search:")
        for i, doc in enumerate(response['context']):
            print(f"Document {i+1}: {doc.page_content}")


if __name__ == "__main__":
    main()
