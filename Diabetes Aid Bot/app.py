import os
import gradio as gr
from dotenv import load_dotenv
from PyPDF2 import PdfReader  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import openai

# ✅ Load environment variables
load_dotenv()

# ✅ Load and process the MealPlans.pdf using PyPDF2
pdf_path = "MealPlans.pdf"

def load_pdf_text(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ✅ Check if the file exists before processing
if os.path.exists(pdf_path):
    meal_plans_text = load_pdf_text(pdf_path)
else:
    raise FileNotFoundError("⚠️ Meal Plans PDF not found!")

# ✅ Initialize OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("⚠️ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")

# ✅ Initialize OpenAI Client (Fixed for OpenAI v1.x API)
client = openai.OpenAI(api_key=openai_api_key)

# ✅ Initialize FAISS Vector Store with Meal Plans
try:
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts([meal_plans_text], embedding=embeddings)
except Exception as e:
    raise Exception(f"⚠️ Error initializing FAISS vector store: {str(e)}")

# ✅ Function to Retrieve Personalized Meal Plans (Fixed Chat History)
def get_diet_plan(user_input, history=[]):
    """Retrieve personalized meal plans based on user input and maintain chat history."""
    try:
        # Retrieve similar meal plans from FAISS
        docs = vector_store.similarity_search(user_input, k=3)
        meal_suggestions = '\n\n'.join([doc.page_content for doc in docs])

        # ✅ Build message history (OpenAI needs full chat history)
        messages = [{"role": "system", "content": "You are a diabetes dietitian providing personalized South Indian meal plans."}]

        # ✅ Add past chat history
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        # ✅ Append the current user query
        messages.append({"role": "user", "content": f"Provide a meal plan based on this query: {user_input}\n\n{meal_suggestions}"})

        # ✅ Fixed OpenAI API Call (Using Correct v1.x Syntax)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )

        bot_response = response.choices[0].message.content

        # ✅ Update chat history
        history.append((user_input, bot_response))

        return bot_response, history

    except Exception as e:
        return f"⚠️ Error generating diet plan: {str(e)}", history

# ✅ 🎨 Fancy Gradio UI with Custom Styling
css = """
h1 {
    text-align: center;
    color: #008000;
    font-size: 2.5rem;
}
body {
    background: linear-gradient(to right, #f3f4f6, #e0f7fa);
}
#chatbot {
    font-size: 1.2rem;
    background: #ffffff;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
}
.gradio-container {
    max-width: 700px;
    margin: auto;
}
"""

# ✅ Gradio Interface (Fixed with Chat History)
with gr.Blocks(css=css) as app:
    gr.Markdown("# 🍛 **DiabetesDietBot** - Meal Planner for Type 2 Diabetes")

    gr.Markdown("👋 Welcome! Enter details like **'Male, 40s, Vegetarian, Moderate Calories'** to get a **customized diabetes-friendly meal plan!**")

    chatbot = gr.ChatInterface(
        fn=get_diet_plan,  # ✅ Uses the updated function with history
        title="DiabetesDietBot",
        chatbot=True
    )

    with gr.Row():
        img = gr.Image("meal_plan_example.jpg", width=400, height=250, interactive=False)
        gr.Markdown("### Example: South Indian Diabetes-Friendly Meal Plan")

    gr.Markdown("### 🔗 Developed by [Dr. Syed Faizan](https://www.linkedin.com/in/drsyedfaizanmd/)")

# ✅ Launch Gradio App
app.launch(share=True)  # ✅ Enables Public Link
