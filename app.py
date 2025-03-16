import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import numpy as np

os.environ["OPENAI_API_KEY"] = openai_key

class IntentClassifier:
    def __init__(self, training_data):
        self.vectorizer = CountVectorizer()
        self.model = SVC(kernel="linear", probability=True)
        self.train(training_data)
    
    def train(self, training_data):
        texts = [item["text"] for item in training_data]
        intents = [item["intent"] for item in training_data]
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, intents)
    
    def predict(self, user_input):
        user_input_vector = self.vectorizer.transform([user_input])
        return self.model.predict(user_input_vector)[0]

class VectorDatabase:
    def __init__(self, documents):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma.from_documents(documents, self.embeddings)
    
    def retrieve(self, query):
        retriever = self.vector_store.as_retriever()
        return retriever.get_relevant_documents(query)

class ChatSystem:
    def __init__(self, model="gpt-4", temperature=0.7):
        self.chat_model = ChatOpenAI(model=model, temperature=temperature)
    
    def get_response(self, query, retrieval_chain):
        return retrieval_chain.run(query)

# Sample training data
training_data = [
    {"text": "How to set up a financial period?", "intent": "Set Financial Period"},
    {"text": "How to configure general system settings?", "intent": "General Settings"},
    {"text": "How to define account coding?", "intent": "Account Coding"},
    {"text": "Tell me about Amazon company.", "intent": "Company Introduction"},
    {"text": "What systems does Amazon software include?", "intent": "Product Introduction"},
]

# Sample documents for vector database
documents = [
    Document(page_content="Financial period setup: After installing the system, you need to set up the desired financial period. 
    Navigate to the basic information section and select the financial period and document numbering option. The system creates a 
    default financial period that you can modify."),
    Document(page_content="General settings: After installation, the default company name is 'Sample Company'. You should update 
    the company name and enter company information."),
    Document(page_content="Account coding settings: You need to define account coding, including group, main, and sub-accounts. 
    It is recommended to use the default standard coding, which automatically generates these accounts."),
    Document(page_content="Amazon is a company that provides financial and accounting software. It offers implementation, 
    support, and software products both locally and internationally."),
    Document(page_content="Amazon software includes accounting, liquidity, sales, tax compliance, payroll, inventory, 
    production, quality control, and e-commerce systems."),
]

# Initialize components
intent_classifier = IntentClassifier(training_data)
vector_db = VectorDatabase(documents)
chat_system = ChatSystem()
retrieval_chain = RetrievalQA.from_chain_type(retriever=vector_db.vector_store.as_retriever(), llm=chat_system.chat_model)

# Run the chatbot
print("System is ready! Type 'exit' to quit.")
while True:
    query = input("Enter your question: ")
    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    intent = intent_classifier.predict(query)
    print(f"Detected Intent: {intent}")
    
    if intent in ["Set Financial Period", "General Settings", "Account Coding"]:
        response = chat_system.get_response(query, retrieval_chain)
        print("Response:", response)
    elif intent == "Company Introduction":
        print("Response: Amazon is a leading company in financial software solutions, offering implementation and support services.")
    elif intent == "Product Introduction":
        print("Response: Amazon software includes accounting, payroll, inventory, production, and other management systems.")
    else:
        print("Response: Sorry, I cannot answer this question.")
