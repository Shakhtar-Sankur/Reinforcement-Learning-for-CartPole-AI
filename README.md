# Reinforcement Learning for CartPole AI

## What is this project?

This project implements a **Deep Q-Network (DQN)** agent to play the **CartPole** game using **Python**, **OpenAI Gym**, and **PyTorch**. The agent learns to balance a pole on a cart through trial-and-error, optimizing its actions to maximize rewards. The project includes training, evaluation, visualization, and an optional web demo, achieving an average reward of **195/200** over 100 test episodes after 1000 training episodes.

### Key components:
* **DQN Algorithm**: Uses a neural network with experience replay and epsilon-greedy exploration to learn optimal actions.
* **Environment**: CartPole from OpenAI Gym, with a 4D state space (position, velocity, angle, angular velocity) and 2 actions (left, right).
* **Tech Stack**: Python, PyTorch, OpenAI Gym, NumPy, Matplotlib, OpenCV, Streamlit.
* **Outputs**: Training plots (`plots/rewards.png`), gameplay video (`videos/gameplay.mp4`), and a Streamlit web demo (`app.py`).

## Why this project?

This project was developed to:
* **Showcase RL Expertise**: Demonstrates proficiency in reinforcement learning, complementing my existing portfolio of AI/ML projects including generative models (DDPM), NLP systems (BERT-based), and deep learning frameworks.
* **Technical Excellence**: Built upon my experience with PyTorch, TensorFlow, and cloud deployment (AWS, GCP) to create an end-to-end ML solution with production-ready features.
* **Quantifiable Results**: Achieves 95% of maximum CartPole reward with optimized hyperparameters, demonstrating the same precision I've applied to previous projects (94.8% accuracy in data leak detection, 92% realism in synthetic medical imaging).
* **Industry Relevance**: Reinforcement learning applications in gaming, robotics, and autonomous systems align with current AI trends and potential internship opportunities in cutting-edge tech companies.

## Results

* **Performance**: Average reward of 195/200 over 100 test episodes.
* **Visualizations**: Training curve in `plots/rewards.png` and gameplay video in `videos/gameplay.mp4`.
* **Demo**: Streamlit app at localhost:8501 showcasing the agent's performance.

## Project Structure

```
rl_game_ai/
├── main.py                 # Training script
├── agent.py                # DQN agent implementation
├── environment.py          # CartPole environment wrapper
├── utils.py                # Helper functions for logging and saving
├── visualize.py            # Generates plots and videos
├── evaluate.py             # Evaluates the trained model
├── app.py                  # Streamlit web demo
├── config.py               # Hyperparameters
├── requirements.txt        # Dependencies
├── models/                 # Saved model files
├── plots/                  # Training visualizations
└── videos/                 # Gameplay recordings
```

## Why it matters

This project demonstrates my ability to implement advanced RL algorithms and handle dynamic environments, building upon my track record of successful AI/ML projects including generative models, NLP frameworks, and IoT predictive systems. It leverages my expertise in PyTorch, cloud deployment, and production-ready ML systems, making it a compelling addition to my portfolio for ML engineering roles and research opportunities.
