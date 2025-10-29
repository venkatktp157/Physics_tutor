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

st.title("🧑‍🏫 Physics Tutor - Dual Agent App")

terminate = st.button("🛑 End Conversation")

if not terminate:
    student_input = st.text_input("👨‍🎓 Student: Ask your question about physics")

    if student_input and not st.session_state.awaiting_answer:
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

            # 👨‍🎓 Ask specific follow-up question
            follow_up_prompt = f"""
            You just explained the following physics concept to a student: {student_input}

            Now ask a specific follow-up question that applies the concept in a real-world scenario.
            Make sure the question is clear and invites the student to reason through the physics.

            Example: "What happens when you slam the brakes while driving?"
            """
            follow_up = chat.invoke([HumanMessage(content=follow_up_prompt)])
            follow_up_text = follow_up.content.strip()

            # Ensure it's not vague
            if "what topic" in follow_up_text.lower() or "how can I assist" in follow_up_text.lower():
                follow_up_text = "Imagine you're driving and suddenly slam the brakes. How do Newton's three laws apply?"

            st.session_state.follow_up_question = follow_up_text
            st.session_state.awaiting_answer = True
            st.markdown("### 👨‍🎓 Follow-Up Question")
            st.write(f"**{follow_up_text}**")

        except Exception as e:
            st.error(f"❌ Error: {e}")

    elif st.session_state.awaiting_answer:
        st.markdown("### 👨‍🎓 Follow-Up Question")
        st.write(f"**{st.session_state.follow_up_question}**")

        student_followup = st.text_input("✍️ Your Answer to the Follow-Up Question")

        if student_followup:
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

                # Reset state
                st.session_state.follow_up_question = None
                st.session_state.awaiting_answer = False

            except Exception as e:
                st.error(f"❌ Evaluation error: {e}")
else:
    st.success("Conversation ended. Thanks for learning physics!")
