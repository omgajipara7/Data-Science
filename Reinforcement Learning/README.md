# 🕹️ Reinforcement Learning (RL)

Reinforcement Learning is a type of machine learning where an agent learns to take actions in an environment to **maximize cumulative rewards**. The agent interacts with the environment, receives feedback (rewards/penalties), and improves its policy over time.

---

## 🧠 Key Concepts

| Concept         | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| Agent           | The learner or decision maker                                               |
| Environment     | The world the agent interacts with                                          |
| Action          | What the agent can do                                                       |
| State           | The current situation of the environment                                    |
| Reward          | Feedback from the environment based on the agent's action                  |
| Policy          | Strategy that the agent uses to decide actions                              |
| Value Function  | Expected long-term return from each state                                   |

---

## 🔁 How It Works

The agent follows this loop:
1. Observe state from the environment.
2. Take an action based on the policy.
3. Receive a reward and new state from the environment.
4. Learn from this experience and update policy.

---

## 🛠️ Popular Algorithms

| Algorithm         | Description                                              |
|------------------|----------------------------------------------------------|
| Q-Learning        | Off-policy algorithm that learns the value of actions   |
| SARSA             | On-policy algorithm that updates using actual taken actions |
| Deep Q-Network (DQN) | Uses neural networks to approximate Q-values         |
| Policy Gradient   | Directly optimizes the policy instead of value functions |

---

## 📂 Folder Structure
- `Q-Learning/`
- `SARSA/`
- `DQN/`
- `PolicyGradient/`

Each folder includes:
- `.ipynb` code
- `index.html` (interactive notebook)
- `README.md` with summary

---

🎮 Explore each algorithm folder to get hands-on with Reinforcement Learning!
