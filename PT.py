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

# 🧠 Ensure session state keys are initialized early
default_state = {
    "memory": ConversationBufferMemory(return_messages=True),
    "follow_up_question": None,
    "awaiting_answer": False,
    "teacher_response": None,
    "student_followup_input": ""
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("🧑‍🏫 Physics Tutor - Dual Agent App")

# 🔄 Clear button to reset and ask a new question
if st.button("🔄 Ask Another Question"):
    for key in default_state.keys():
        st.session_state[key] = default_state[key]
    st.success("Ready for your next question!")

# 🧑‍🎓 Student asks a physics question
if not st.session_state.awaiting_answer:
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
            # 🧑‍🏫 Teacher's explanation
            response = chat.invoke([HumanMessage(content=teacher_prompt)])
            st.session_state.teacher_response = response.content
            st.markdown("### 🧑‍🏫 Teacher's Response")
            st.write(response.content)

            # 🔍 Extract follow-up question
            for line in response.content.splitlines():
                if "Follow-up question:" in line:
                    candidate = line.replace("Follow-up question:", "").strip()
                    if len(candidate) > 10 and "what topic" not in candidate.lower():
                        st.session_state.follow_up_question = candidate
                    break

            # 🧠 Fallback if not found or too vague
            if not st.session_state.follow_up_question:
                fallback_prompt = f"""
                You just explained this physics concept: "{student_input}"

                The teacher forgot to include a meaningful follow-up question. Please generate one that applies this concept in a real-world scenario and invites the student to reason through it.
                """
                fallback_response = chat.invoke([HumanMessage(content=fallback_prompt)])
                st.session_state.follow_up_question = fallback_response.content.strip()

            # 🎨 Extract diagram prompt
            diagram_prompt = f"Physics diagram illustrating: {student_input}"  # default
            for line in response.content.splitlines():
                if "Simple diagram:" in line:
                    diagram_prompt = line.replace("Simple diagram:", "").strip()
                    break

            # 🖼️ Try image generation
            with st.spinner("Generating visual explanation..."):
                try:
                    image_response = openai.images.generate(
                        model="dall-e-3",
                        prompt=diagram_prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1
                    )
                    image_url = image_response.data[0].url
                    st.image(image_url, caption="Visual Explanation")
                except Exception as img_error:
                    st.warning(f"⚠️ Image generation skipped: {img_error}")

            st.session_state.awaiting_answer = True

        except Exception as e:
            st.error(f"❌ Error: {e}")

# 👨‍🎓 Student answers the follow-up question
if st.session_state.awaiting_answer and st.session_state.follow_up_question:
    st.markdown("### 👨‍🎓 Follow-Up Question")
    st.write(f"**{st.session_state.follow_up_question}**")

    student_followup = st.text_input("✍️ Your Answer to the Follow-Up Question", key="student_followup_input")

    if st.button("✅ Submit Answer"):
        if student_followup.strip():
            try:
                evaluation_prompt = f"""
                You are a physics teacher. A student answered your follow-up question.

                Follow-up question: {st.session_state.follow_up_question}
                Student's answer: {student_followup}

                Please:
                1. Evaluate the student's answer.
                2. Provide the correct answer with a brief explanation.
                3. Offer encouragement or a tip for improvement.
                """

                evaluation = chat.invoke([HumanMessage(content=evaluation_prompt)])
                st.markdown("### 🧑‍🏫 Teacher's Evaluation")
                st.write(evaluation.content)

                st.success("✅ Evaluation complete. You can now ask another question using the 🔄 button above.")

            except Exception as e:
                st.error(f"❌ Evaluation error: {e}")
        else:
            st.warning("⚠️ Please enter an answer before submitting.")
