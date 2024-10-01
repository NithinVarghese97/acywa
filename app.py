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
        self.is_new_user = None  # Track if it's a new user, start with None

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
            ("system", "You are a helpful assistant. Help users navigate the Atlas map based on {context}."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", "Be concise, helpful, and clear. Focus only on {context}.")
        ])

        # Create the chain that will process documents and responses
        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        retriever = self.vectorStore.as_retriever(search_kwargs={"k": 1})

        # Define prompt template for retrieval
        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", f"Given the above conversation about {self.context}, generate a search query to look up relevant information.")
        ])

        # Create a history-aware retriever for contextual searches
        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        # Return the retrieval chain that processes user input
        return create_retrieval_chain(
            history_aware_retriever,
            chain
        )

    # Function to guide new users
    def handle_new_user(self):
        instructions = (
            "Great! Let's start by familiarising you with the map platform.\n"
            "You can start by reading the help screens. Please follow these steps:\n"
            "1. Click on Atlas maps.\n"
            "2. Navigate to the right-hand side pane.\n"
            "3. Click the 'i' icon in the top right-hand corner.\n"
            "This will open the help screens. There are three screens covering different aspects of the platform: the National scale, Atlas menu items, and map interactions."
        )
        return instructions

    # Truncate chat history to avoid exceeding token limits
    def truncate_chat_history(self, history, max_tokens):
        total_tokens = 0
        truncated_history = []

        # Traverse the history from most recent to oldest, adding up tokens
        for message in reversed(history):
            token_count = len(message.content.split())  # Roughly estimate tokens based on word count
            if total_tokens + token_count > max_tokens:
                break
            truncated_history.append(message)
            total_tokens += token_count

        # Return the history in the original order (most recent last)
        return list(reversed(truncated_history))

    # Function to process the chat and generate responses
    def process_chat(self, question):
        # Check if the user is new
        if self.is_new_user is None:
            if question.lower() in ['yes', 'y']:
                self.is_new_user = True
                return self.handle_new_user()
            elif question.lower() in ['no', 'n']:
                self.is_new_user = False
                return "Welcome back! What can I assist you with today?"
            else:
                return "Are you new to our interactive map platform? (Yes/No)"

        # Truncate chat history to avoid exceeding token limits
        truncated_history = self.truncate_chat_history(self.chat_history, self.MAX_HISTORY_TOKENS)
        
        # Invoke the chain with the truncated chat history
        response = self.chain.invoke({
            "input": question,
            "chat_history": truncated_history,
            "context": self.context  # Pass context to the chain
        })
        
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    # Function to reset chat history (for new users or conversation reset)
    def reset_chat_history(self):
        self.chat_history = []
        self.is_new_user = None  # Reset the user state

# Map-specific assistant class (inheriting from Assistant)
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
