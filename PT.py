#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
import openai
import warnings
warnings.filterwarnings('ignore')

# ✅ Load API keys from secrets.toml
groq_api_key = st.secrets["GROQ_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ❗ Safety check
if not groq_api_key or not openai.api_key:
    st.error("❌ Missing API keys in secrets.toml. Please add GROQ_API_KEY and OPENAI_API_KEY.")
    st.stop()

# 🚀 Initialize Groq model
chat = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# 🧠 Setup memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

st.title("🧑‍🏫 Physics Tutor - Dual Agent App")

terminate = st.button("🛑 End Conversation")

if not terminate:
    student_input = st.text_input("👨‍🎓 Student: Ask your question about physics")

    if student_input:
        st.session_state.memory.chat_memory.add_user_message(student_input)

        teacher_prompt = f"""
        You are a physics teacher helping a student understand physics concepts.

        For the student's question, respond with:
        1. A clear and concise explanation of the concept.
        2. A real-world analogy or example.
        3. A simple diagram or image description that could help visualize the concept.
        4. Common misconceptions or mistakes students make.
        5. A follow-up question to deepen understanding.

        Question: {student_input}
        """

        try:
            # 🧑‍🏫 Get teacher response
            response = chat.invoke([HumanMessage(content=teacher_prompt)])
            st.markdown("### 🧑‍🏫 Teacher's Response")
            st.write(response.content)

            # 🖼️ Generate image based on student question
            with st.spinner("Generating visual explanation..."):
                image_prompt = f"Physics diagram illustrating: {student_input}"
                image_response = openai.Image.create(
                    prompt=image_prompt,
                    n=1,
                    size="512x512"
                )
                image_url = image_response['data'][0]['url']
                st.image(image_url, caption="Visual Explanation")

            # 👨‍🎓 Ask follow-up question
            follow_up = chat.invoke([HumanMessage(content="Ask the student a follow-up question to reinforce learning.")])
            st.markdown("### 👨‍🎓 Student's Turn")
            st.write(follow_up.content)

        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.success("Conversation ended. Thanks for learning physics!")
