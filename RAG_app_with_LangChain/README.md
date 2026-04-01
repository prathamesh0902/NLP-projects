# Building-RAG-App-With-LangChain

[Link](https://www.youtube.com/watch?v=seSjDT846LA&t=752s)

Test Questions for your RAG Pipeline for `box_data.txt` flie:
- **Fact Check:** "In which room can I find the $9,500 Black Box?"
- **Inference:** "What is the cheapest loose item available in the Orange Room?"
- **Calculation:** "If I buy a dozen Alphonso Mangoes and a Pink Box from the Black Room, how much will I spend?"

`python -m venv pjvenv`
`pip install -r requirements.txt`
`streamlit run app.py`


GOOGLE_API_KEY from Google AI Studio (daggerknight99@gmail.com)


**Project Overview**
1. LLM as ChatGoogleGenerativeAI
2. Embeddings as GoogleGenerativeAIEmbeddings
3. Load file in pdf / docx / txt format
4. Vectorstore as FAISS
5. RetrievalQA
6. Upload file and ask questions
7. Build streamlit app
