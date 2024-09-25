from flask import Flask, request, jsonify
import openai
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever

# Load environment variables (e.g., API keys)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend interaction

# Set OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is not set in the environment variables.")
openai.api_key = openai_api_key

# Function to load text file
def load_text(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    
    loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

# Function to create vector store for document embeddings
def create_db(docs):
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return Chroma.from_documents(docs, embedding=embedding)

# Function to create the LangChain retrieval chain
def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.4,
        api_key=openai_api_key
    )

    # Define prompt for generating responses
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}. Start the conversation with: If given incomplete questions, e.g., one-word input, please ask follow-up questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Create chain to process documents and context
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retrieval_chain

# Function to process chat with LangChain and the retrieval chain
def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]

# Root route to handle homepage
@app.route('/')
def index():
    return "Welcome to the Chatbot API! Access the /chat endpoint to communicate with the chatbot."

# Chat route to handle incoming chat requests
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    chat_history = data.get("chat_history", [])

    if user_message:
        try:
            # Load documents and set up vector store (for demo purposes; adjust as needed)
            text_docs = load_text('Maps Prompts.txt')
            all_docs = text_docs
            vectorStore = create_db(all_docs)
            chain = create_chain(vectorStore)
            
            # Process the user message through LangChain
            bot_reply = process_chat(chain, user_message, chat_history)
            
            # Append the messages to chat history
            chat_history.append(HumanMessage(content=user_message))
            chat_history.append(AIMessage(content=bot_reply))

            return jsonify({"reply": bot_reply, "chat_history": chat_history})
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"reply": "Sorry, there was an error processing your request."}), 500

    return jsonify({"reply": "No message provided."}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
