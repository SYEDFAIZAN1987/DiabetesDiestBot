---
title: Diabetes_Diet_Bot
app_file: app.py
sdk: gradio
sdk_version: 5.12.0
---
# 🍛 **DiabetesDietBot** - South Indian Meal Planner for Type 2 Diabetes

## **Introduction**
DiabetesDietBot is an **AI-powered chatbot** designed to provide **personalized meal plans** for individuals with **Type 2 Diabetes**, focusing on **South Indian dietary preferences**. This chatbot leverages **Retrieval-Augmented Generation (RAG)** technology to retrieve relevant **meal plans** from a structured knowledge base (`MealPlans.pdf`), ensuring customized, nutritionally balanced recommendations.

---

## **🎯 Features**
✔ **Personalized Meal Plans** – Based on **age, gender, dietary preference, and caloric needs**  
✔ **South Indian Focus** – Meal plans aligned with traditional **vegetarian, non-vegetarian, and non-onion/garlic** diets  
✔ **AI-Powered Recommendations** – Uses **FAISS indexing and OpenAI GPT** to provide the most relevant meal plans  
✔ **Gradio-Based Chat Interface** – User-friendly, mobile-friendly chatbot for instant diet advice  
✔ **Fast Retrieval** – FAISS-based vector search enables quick meal plan suggestions  

---

## **🛠 Technologies Used**
### **Backend:**
- **LangChain** – RAG-based retrieval and AI-assisted meal planning
- **FAISS** – Vector search for quick and relevant meal plan retrieval
- **OpenAI GPT API** – AI-powered chatbot responses

### **Frontend:**
- **Gradio** – Interactive chatbot UI with a sleek, modern interface

### **Data Source:**
- **MealPlans.pdf** – Contains 40+ structured meal plans tailored for Type 2 Diabetes in South Indian contexts

---

## **📂 Project Files**
- `DiabetesDietBot.py` → Main chatbot script (Gradio-based UI)
- `rag.py` → Text processing & vector indexing with FAISS
- `rag2.py` → Alternate FAISS indexing method
- `MealPlans.pdf` → Core dataset for AI meal plan recommendations
- `requirements.txt` → Dependencies for Hugging Face deployment

---

## **🚀 Deployment on Hugging Face**
You can use **Hugging Face Spaces** to deploy DiabetesDietBot:

### **1️⃣ Manual Deployment**
1. **Go to [Hugging Face Spaces](https://huggingface.co/spaces)**
2. Click **"New Space"** → Choose **Gradio** as the SDK  
3. **Upload the following files manually:**
   - `DiabetesDietBot.py`
   - `rag.py`
   - `rag2.py`
   - `MealPlans.pdf`
   - `requirements.txt`
4. **Restart Space** and your chatbot will be live!

---

## **🔧 Local Installation & Running**
To run the chatbot locally:

### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
