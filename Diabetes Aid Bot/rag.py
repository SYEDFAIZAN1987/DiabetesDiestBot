# %% Packages
import os
import re
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai

# Load environment variables from .env file
load_dotenv()

# Ensure OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("⚠️ OpenAI API Key is missing. Set it in your .env file or environment variables.")

# Create OpenAI client (✅ Fix for v1.x API)
client = openai.OpenAI(api_key=openai_api_key)

# Load the Meal Plans PDF
pdf_path = "MealPlans.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError("⚠️ Meal Plans PDF not found!")

# Extract text from PDF
reader = PdfReader(pdf_path)
meal_texts = [page.extract_text().strip() for page in reader.pages if page.extract_text()]

# Clean extracted text
cleaned_texts = [re.sub(r'\d+\n.*?\n', '', text) for text in meal_texts]

# Split text into chunks
char_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=500,
    chunk_overlap=50
)

# Ensure `document_chunks` is a **flat list**
document_chunks = char_splitter.split_text("\n\n".join(cleaned_texts))  # Convert to a single string first
print(f"✅ Number of text chunks: {len(document_chunks)}")

# Generate embeddings
embeddings = OpenAIEmbeddings()

# Ensure FAISS receives a list of strings (not nested lists)
vector_store = FAISS.from_texts(document_chunks, embedding=embeddings)

# Save FAISS index
vector_store.save_local("db_mealplans")
print("✅ FAISS index successfully created and saved for DiabetesDietBot.")

# Define RAG Query Function
def rag(query, n_results=5):
    """Retrieve personalized meal plans based on dietary preferences."""
    try:
        # Query FAISS vector store
        docs = vector_store.similarity_search(query, k=n_results)
        retrieved_text = "; ".join([doc.page_content for doc in docs])

        # Prepare AI prompt
        messages = [
            {"role": "system", "content": "You are a diabetes dietitian specializing in South Indian meal planning."},
            {"role": "user", "content": f"Generate a personalized meal plan: {query}\n\n{retrieved_text}"}
        ]

        # ✅ Fixed OpenAI API Call (New v1.x API)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error generating diet plan: {str(e)}"

# Example Query Test
if __name__ == "__main__":
    test_query = "Male, 50s, Vegetarian, Moderate Calories"
    response = rag(query=test_query, n_results=5)
    print("\n🛑 Sample Response:\n")
    print(response)
