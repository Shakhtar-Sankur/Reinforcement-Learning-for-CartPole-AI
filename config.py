import streamlit as st
import base64
import os

def get_video_base64(video_path):
    """Convert video file to base64 for embedding in Streamlit."""
    with open(video_path, "rb") as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode()

st.title("CartPole DQN Demo")
st.write("Reinforcement Learning agent trained to play CartPole.")

# Display training plot
st.image("plots/rewards.png", caption="Training Rewards vs. Episodes", use_column_width=True)

# Display gameplay video
video_path = "videos/gameplay.mp4"
if os.path.exists(video_path):
    video_base64 = get_video_base64(video_path)
    st.video(f"data:video/mp4;base64,{video_base64}")
else:
    st.write("Gameplay video not found. Run visualize.py to generate it.")