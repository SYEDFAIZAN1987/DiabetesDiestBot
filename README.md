# 🍛 DiabetesDietBot - AI-Based Dietary Assistant for Type 2 Diabetes
![Diabetes Diet Bot](https://github.com/SYEDFAIZAN1987/First-Aid-Tutor/blob/main/RAGGPT%20UI.png)
## 📌 Overview
**DiabetesDietBot** is an AI-powered chatbot developed by **Syed Faizan** for **Mysore Medical College and Research Institute** as part of a research study on how chatbots can assist **doctors and medical students** in recommending **predefined diet plans** to patients with **Type 2 Diabetes**. 

The bot leverages **Retrieval-Augmented Generation (RAG)**, **FAISS vector search**, and **OpenAI's GPT-3.5-turbo** to provide **personalized diet recommendations** based on patient profiles, such as:
- **Age** (e.g., 40s, 50s)
- **Dietary Preferences** (e.g., Vegetarian, Non-Vegetarian)
- **Caloric Requirements** (e.g., Low, Moderate, High)

It uses a dataset of **structured South Indian meal plans** extracted from a **PDF database** to generate optimized diet recommendations.

---

## 🚀 Features
- ✅ **Personalized Diet Plans** – Generates meal plans tailored to user inputs.
- ✅ **RAG-Based Search** – Retrieves relevant meal plans using FAISS for similarity search.
- ✅ **Gradio UI** – Interactive chatbot interface for ease of use.
- ✅ **Secure OpenAI API Integration** – Uses GPT-3.5-turbo for intelligent responses.
- ✅ **Customizable Meal Database** – Works with preloaded **PDF meal plans**.
- ✅ **Optimized for South Indian Diets** – Designed for diabetes-friendly traditional meals.

---

## 📂 Project Structure
```
📦 DiabetesDietBot
├── 📜 app.py # Main Gradio-based chatbot UI
├── 📜 rag.py # Retrieval-Augmented Generation (RAG) implementation
├── 📜 rag2.py # Alternative FAISS-based text processing
├── 📂 db_mealplans # FAISS vector store for meal plan retrieval
├── 📜 .env # Environment variables (OpenAI API key)
└── 📜 README.md # Project documentation (this file)
```

## ⚙️ Setup Instructions

### 1️⃣ **Clone the Repository**
```
git clone https://github.com/your-repo/DiabetesDietBot.git
cd DiabetesDietBot
```

### 2️⃣ **Install Dependencies**  
Ensure you have **Python 3.8+** installed. Then run:  

```
pip install -r requirements.txt
```

### 3️⃣ **Set Up API Keys**  
Create a `.env` file in the project root and add your **OpenAI API key**:  

```
OPENAI_API_KEY=your_openai_api_key
```
## 🛠️ Technical Details  

### 1️⃣ **Text Extraction from PDFs**  
The meal plans are extracted from `MealPlans.pdf` using `PyPDF2`:  

```
from PyPDF2 import PdfReader
reader = PdfReader("MealPlans.pdf")
text = "\n".join([page.extract_text() for page in reader.pages])
```

### 2️⃣ **FAISS-Based Retrieval**  
The extracted text is **embedded** using `OpenAIEmbeddings` and stored in a **FAISS vector database**:  

```
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts([meal_plans_text], embedding=embeddings)
```
### 3️⃣ **Retrieval-Augmented Generation (RAG)**  
When a user inputs a query, FAISS retrieves **top 3** most relevant meal plans:  

```
docs = vector_store.similarity_search(user_input, k=3)
meal_suggestions = '\n\n'.join([doc.page_content for doc in docs])
```


---

## 4️⃣ Gradio Chatbot UI  
A user-friendly **Gradio-based chatbot interface** allows interaction


---

## 🎨 Gradio UI Preview  
The chatbot has a **minimalist UI** with meal plan visualization.

Users can input preferrences
And receive a **tailored meal plan** optimized for **Type 2 Diabetes**.

---

## 📅 Future Enhancements  
- 🔹 **Expand Meal Database** – Include more regional & international diabetes diets.  
- 🔹 **Nutritional Analysis** – Provide macronutrient breakdown for recommended meals.  
- 🔹 **Voice Interface** – Integrate with **speech recognition** for accessibility.  
- 🔹 **Doctor's Portal** – Allow medical professionals to curate and modify diet plans.  

---

## 📜 License  
This project is **open-source** and licensed under the **MIT License**.

---

## 👨‍⚕️ Developed By  
🛠 **Dr. Syed Faizan**  
📍 **Mysore Medical College & Research Institute**  
🔗 [LinkedIn](https://www.linkedin.com/in/drsyedfaizanmd/)  

For collaborations or research inquiries, feel free to reach out!













