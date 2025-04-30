# RAG-Prompt-Optimizer

This project is a document-based **Retrieval-Augmented Generation (RAG)** system that improves user-provided prompts using **Google Gemini** and a custom **PDF knowledge base**. It retrieves relevant content from documents and uses it to generate high-quality, optimized prompts.

---

## 🚀 Features

- 🔎 Context-aware prompt enhancement
- 📄 Document-based knowledge using PDF inputs
- 🤖 Powered by **Gemini 1.5 Flash**
- 📚 Embedding via `GoogleGenerativeAIEmbeddings`
- 🧠 Vector search with **ChromaDB**
- 🔧 Fully local pipeline except for Gemini API calls

---

## 📂 Project Structure

- Loads a PDF file (reader.pdf) and splits it into chunks.
- Converts the chunks into vector embeddings using Google Generative AI embeddings.
- Stores them in a Chroma vector DB.
- When a user types a prompt, it retrieves the most relevant document chunks.
- Feeds those chunks + the prompt to Gemini, which returns an optimized version.

## Updates

- Dynamic Question Generation:
*The prompt template now instructs the LLM to either return an optimized prompt or generate follow-up questions when context is insufficient*

