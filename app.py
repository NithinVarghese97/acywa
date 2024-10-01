import os
import openai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
 
# Class for assistant chatbot
class Assistant:
    MAX_HISTORY_TOKENS = 2000  # Adjust the token limit as necessary
 
    def __init__(self, file_path, context):
        self.context = context
        self.docs = self.load_text(file_path)
        self.vectorStore = self.create_db(self.docs)
        self.chain = self.create_chain()
        self.chat_history = []
        self.is_new_user = False  # Track whether it's a new user or not
 
    # Function to load the text document
    def load_text(self, file_path):
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()
 
    # Function to create the vector store using embeddings
    def create_db(self, docs):
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return Chroma.from_documents(docs, embedding=embedding)
 
    # Function to create the retrieval chain with your specific prompt instructions
    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4-turbo",  # Adjust the model as needed
            temperature=0.4,
            api_key=openai_api_key
        )
 
        # Custom prompt logic with a simplified system prompt to reduce token usage
        prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant chatbot specifically focused on giving a tutorial on how to navigate the Atlas map, based on {context}. "
               "Your primary goal is to help users with {context} only."),
    ("system", "Context: {context}"),
    ("system", "Instructions for {context}:"
               "\n1. If given a one-word or vague query, ask for clarification before proceeding."
               "\n2. For all users, provide the following general steps for finding data on a specific theme or indicator:"
               "\n   - Direct users to open the Atlas maps"
               "\n   - Instruct users to use the theme or indicator search box in Atlas maps"
               "\n   - Explain that if data is available on the topic, it will appear as a dropdown"
               "\n   - Do not interpret specific data or findings"
               "\n3. Always relate your responses back to the user's original query, regardless of the theme or indicator."),
        MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Remember to be concise, clear, and helpful in your responses - give a maximum of 3 sentences. "
               "Focus exclusively on {context} and do not discuss other topics unless explicitly asked."
               "After giving guidance, suggest two relevant follow-up questions.")
        ])
 
        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )
        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})
 
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", f"Given the above conversation about {self.context}, generate a search query to look up relevant information")
        ])
 
        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )
 
        return create_retrieval_chain(
            history_aware_retriever,
            chain
        )
 
    def process_chat(self, question):
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context
        })
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]
 
    def reset_chat_history(self):
        self.chat_history = []
 
class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('Raw data - maps.txt', 'map navigation')
 
# Route to handle homepage (Flask API)
@app.route('/')
def index():
    return "Welcome to the Atlas Map Navigation Assistant API! Use /chat to interact."
 
# Chat route to handle user interactions (Flask API)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    chat_history = data.get("chat_history", [])
 
    # Initialize the assistant (in this case, map-specific)
    assistant = MapAssistant()
 
    if user_message:
        try:
            # Process the user message through the assistant
            bot_reply = assistant.process_chat(user_message)
            # Append to chat history
            chat_history.append(HumanMessage(content=user_message))
            chat_history.append(AIMessage(content=bot_reply))
 
            # Serialize chat history for frontend or API client use
            serialized_history = [{"type": "human", "content": user_message}, {"type": "ai", "content": bot_reply}]
 
            return jsonify({"reply": bot_reply, "chat_history": serialized_history})
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"reply": "An error occurred while processing your request."}), 500
 
    return jsonify({"reply": "No message provided."}), 400
 
 
# Command-line Interactive Mode
def start_console_chat():
    assistant = MapAssistant()
 
    print("Hello! Welcome to the Atlas Map Navigation Assistant! Are you new to our interactive map platform? (Yes/No)")
 
    user_response = input("You: ").lower()
    if user_response in ['yes', 'y']:
        assistant.is_new_user = True
        print("Great! Let's start by familiarising you with the map platform.")
        print("You can start by reading the help screens. Please follow these steps:")
        print("1. Click on Atlas maps")
        print("2. Navigate to the right-hand side pane")
        print("3. Click the 'i' icon in the top right-hand corner")
        print("This will open the help screens. There are three screens covering different aspects of the platform: the National scale, Atlas menu items, and map interactions.")
        print("Are you ready to continue? (Yes/No)")
        continue_response = input("You: ").lower()
        if continue_response in ['yes', 'y']:
            print("Great! What specific question can I assist you with first?")
        else:
            print("Alright. Feel free to ask any questions when you're ready to explore further.")
    else:
        print("Welcome back! I'm here to assist you with any questions about our map platform. What can I help you with today?")
 
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
 
        try:
            response = assistant.process_chat(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try that again. Could you rephrase your question?")
 
 
# Entry point for either Flask API or console mode
if __name__ == '__main__':
    mode = input("Choose mode (api/console): ").strip().lower()
    if mode == "api":
        # Run the Flask app
        app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    elif mode == "console":
        # Run the console-based chat
        start_console_chat()
    else:
        print("Invalid mode. Choose either 'api' or 'console'.")
