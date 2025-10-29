# 🇫🇷 Physics Tutor — Streamlit App Powered by Groq

An educational dual-agent chatbot for learning Physics. One agent asks grammar questions in Physics, and the other responds with proper explanations in English — powered by Groq's high-performance LLM.

## 🚀 Features

- AI teacher trained to answer Physics queries
- Translates responses into English
- Looping Q&A until terminated manually
- Streamlit UI + Groq backend via LangChain

## 🧰 Tech Stack

- Streamlit
- Groq LLM via LangChain (`llama-3.3-70b-versatile`)
- Python
- `secrets.toml` for secure key loading

## 📦 Installation

```bash
git clone https://github.com/yourusername/FrenchGrammarTutor.git
cd Physics_tutor

# Optional: create environment
python -m venv tutor_env
tutor_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
