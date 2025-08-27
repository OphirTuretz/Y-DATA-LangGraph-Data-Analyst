# 🤖 Y-DATA LangGraph Data Analyst Agent

Final project for the **Agentic Systems** course, part of the **Y-DATA 2024–2025** program.  
This repository contains an interactive **LangGraph-based data analysis agent** that can answer user questions about the [Bitext – Customer Service Tagged Training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).  

It extends the first assignment (basic ReAct agent) by re-implementing it in **LangGraph** and adding **conversation memory, persistence, and a corresponding Streamlit interface**.

---

## ✨ Features

- 🧭 **Router Node** – classifies queries into:  
  - Structured  
  - Unstructured  
  - Memory  
  - Out-of-scope

- 📊 **Structured Query Agent** – supports:  
  - Counting categories or intents  
  - Showing examples  
  - Getting distributions  
  - Sorting dictionaries, summing values, etc.

- 📝 **Unstructured Query Agent** – performs **summarization across dataset batches** with dedicated prompts.

- 🚫 **Out-of-scope Handler** – returns polite fallback responses.

- 🧠 **Summarized Memory Node** – stores & retrieves **concise user memories** across sessions.

- 💾 **Persistence** –  
  - User & thread IDs, histories stored in **SQLite**  
  - **LangGraph checkpoints** for conversation state  
  - **SQLite store** for summarized memory

- 💻 **Streamlit Interface** –  
  - User & thread management  
  - Conversation history view  
  - Query submission & results

---

## ⚙️ Tech Stack

- [LangChain](https://python.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/) – agent orchestration  
- [OpenAI GPT-4o-mini](https://platform.openai.com/) – LLM backbone  
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/) – Bitext dataset  
- [Streamlit](https://streamlit.io/) – web UI  
- [SQLite](https://www.sqlite.org/) – persistence (checkpoints, store, user/thread DBs)  
- [Pydantic](https://docs.pydantic.dev/) – structured outputs  
- [dotenv](https://pypi.org/project/python-dotenv/) – environment variables

---

## 🚀 Installation & Setup

Tested with **Python 3.13.7**.

1. **Clone the repository**
   ```bash
   git clone https://github.com/OphirTuretz/Y-DATA-LangGraph-Data-Analyst.git
   cd Y-DATA-LangGraph-Data-Analyst
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Open .env and set your OpenAI API key:
   # OPENAI_API_KEY=sk-...
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run DataAnalyst.py
   ```

6. **(Optional) Reset the environment**
   ```bash
   python cleanup.py
   ```

---

## 🧠 LangGraph Architecture

<p align="center">
  <img src="images/Architecture_Diagram.svg" alt="LangGraph workflow diagram" />
</p>

- **Router Node** → classifies the query.  
- **Structured Agent** ↔ **Structured Tools** (loop until completion).  
- **Unstructured Agent** ↔ **Unstructured Tools** (loop until completion).  
- **Out-of-scope Handler** → returns polite response.  
- **Memory Nodes** – `save_memory` after each turn, `read_memory` when asked.  
- **Checkpointer + Store** – ensure state and memories persist.  

---

## 📂 Repository Structure

```
.
├── DataAnalyst.py              # Streamlit front-end
├── engine.py                   # Query processing wrapper
├── graph.py                    # LangGraph workflow definition
├── graph_state.py              # Shared state schema
├── structured_query_agent.py   # Structured agent + tools
├── unstructured_query_agent.py # Unstructured agent + summarization
├── out_of_scope_query_handler.py
├── summarized_memory.py        # Save/read memory nodes
├── id_manager.py               # User & thread persistence (SQLite)
├── data.py                     # Dataset wrapper
├── general_tools.py            # Shared tools
├── cleanup.py                  # Utility to reset DBs
├── app/const.py                # Config & constants
├── prompts/                    # System prompt templates
├── images/                     # Diagrams
├── requirements.txt
└── .env.example
```

---

## 💡 Example Queries

**Structured**
- *What are the most frequent categories?*  
- *Show 3 examples from the REFUND category*  

**Unstructured**
- *Summarize how agents respond to payment issues*  

**Memory**
- *What do you remember about me?*  

**Out-of-scope**
- *Who is Magnus Carlsen?*  

---

## 📖 Notes

- Assignment: **Final project in Agentic Systems course (Y-DATA 2025)**  
- Python version: **3.13.7**  
- Dataset: [Bitext – Customer Service Tagged Training](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)  
- LangSmith integration available but optional (see `.env.example`)  