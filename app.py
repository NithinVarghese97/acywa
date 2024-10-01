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

    # Function to create the retrieval chain with prompt template
    def create_chain(self):
        model = ChatOpenAI(
            model="gpt-4-turbo",  # Adjust the model as needed
            temperature=0.4,
            api_key=openai_api_key
        )

        # Define the prompt template for responses, now including `context`
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful AI assistant chatbot focused on {self.context}. "
                       f"Your primary goal is to help users navigate the platform."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("system", f"Please ensure that you give concise, clear responses based on the context of {self.context}. "
                       "Provide a maximum of three sentences, and suggest two follow-up questions."),
            # Add context to the prompt
            ("system", "{context}")
        ], input_variables=["chat_history", "input", "context"])

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

    # Function to process the chat and generate responses
    def process_chat(self, question):
        response = self.chain.invoke({
            "input": question,
            "chat_history": self.chat_history,
            "context": self.context  # Pass context to the chain
        })
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=response["answer"]))
        return response["answer"]

    # Function to reset chat history (for new users or conversation reset)
    def reset_chat_history(self):
        self.chat_history = []

# Map-specific assistant class (inheriting from Assistant)
class MapAssistant(Assistant):
    def __init__(self):
        super().__init__('Raw data - maps.txt', 'map navigation')

# Route to handle homepage
@app.route('/')
def index():
    return "Welcome to the Atlas Map Navigation Assistant API! Use /chat to interact."

# Chat route to handle user interactions
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

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
