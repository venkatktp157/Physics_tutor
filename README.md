# 🇫🇷 French Grammar Tutor — Streamlit App Powered by Groq

An educational dual-agent chatbot for learning French grammar. One agent asks grammar questions in French, and the other responds with proper explanations and English translations — powered by Groq's high-performance LLM.

## 🚀 Features

- AI teacher trained to answer French grammar queries
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
cd FrenchGrammarTutor

# Optional: create environment
python -m venv tutor_env
tutor_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
