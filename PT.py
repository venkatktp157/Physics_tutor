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

# 🧠 Setup memory and state
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if "follow_up_question" not in st.session_state:
    st.session_state.follow_up_question = None
if "awaiting_answer" not in st.session_state:
    st.session_state.awaiting_answer = False
if "teacher_response" not in st.session_state:
    st.session_state.teacher_response = None

st.title("🧑‍🏫 Physics Tutor - Dual Agent App")

# 🔄 Clear button to reset and ask a new question
if st.button("🔄 Ask Another Question"):
    for key in ["follow_up_question", "awaiting_answer", "teacher_response", "student_followup_input"]:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_set_query_params()  # optional: clears URL state
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

            # 🖼️ Try image generation
            with st.spinner("Generating visual explanation..."):
                try:
                    image_prompt = f"Physics diagram illustrating: {student_input}"
                    image_response = openai.images.generate(
                        model="dall-e-3",
                        prompt=image_prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1
                    )
                    image_url = image_response.data[0].url
                    st.image(image_url, caption="Visual Explanation")
                except Exception as img_error:
                    st.warning(f"⚠️ Image generation skipped: {img_error}")

            # Extract follow-up question from teacher response
            for line in response.content.splitlines():
                if "Follow-up question:" in line:
                    st.session_state.follow_up_question = line.replace("Follow-up question:", "").strip()
                    break

            # Fallback if not found
            if not st.session_state.follow_up_question:
                st.session_state.follow_up_question = "Imagine you're driving and suddenly slam the brakes. How do Newton's three laws apply?"

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

                # Show clear button to proceed
                st.success("✅ Evaluation complete. You can now ask another question using the 🔄 button above.")

            except Exception as e:
                st.error(f"❌ Evaluation error: {e}")
        else:
            st.warning("⚠️ Please enter an answer before submitting.")
