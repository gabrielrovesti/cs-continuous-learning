# Complete Course Notes
## Deep Reinforcement Learning, Convolutional Q-Learning, A3C, and Large Language Models

---

# Part 0 — Fundamentals of Reinforcement Learning

## 1. Deep Learning Fundamentals: Neural Networks & Activation Functions

A **neural network** is a parameterized function approximator composed of stacked layers of artificial neurons. Each neuron computes:

```
output = activation(W · x + b)
```

where `W` is the weight matrix, `x` the input vector, and `b` the bias.

### Layer Types
- **Input layer**: encodes the state or observation.
- **Hidden layers**: learn intermediate representations.
- **Output layer**: produces a prediction, value, or action distribution.

### Activation Functions

| Function | Formula | Range | Notes |
|---|---|---|---|
| Threshold (step) | `1 if x ≥ 0 else 0` | {0,1} | Non-differentiable; unusable in backprop |
| Sigmoid | `1 / (1 + e^{-x})` | (0, 1) | Squashes output; vanishing gradient at extremes |
| Tanh | `(e^x - e^{-x}) / (e^x + e^{-x})` | (−1, 1) | Zero-centered; still has vanishing gradient |
| ReLU | `max(0, x)` | [0, +∞) | Sparse activation; dead neuron problem |
| Leaky ReLU | `x if x > 0 else αx` | (−∞, +∞) | Fixes dead neurons with small slope α |
| Softmax | `e^{x_i} / Σ e^{x_j}` | (0,1), sums to 1 | Used in output layer for probability distributions |

ReLU is the standard choice for hidden layers in deep RL due to computational efficiency and gradient flow properties.

---

## 2. How Reinforcement Learning Works

### The RL Framework

Reinforcement Learning models an agent interacting with an environment over discrete time steps. At each step `t`:

1. Agent observes state `s_t`.
2. Agent selects action `a_t` according to policy `π(a | s)`.
3. Environment transitions to `s_{t+1}` and emits reward `r_t`.

The agent's objective is to maximize the **expected cumulative discounted return**:

```
G_t = Σ_{k=0}^{∞} γ^k · r_{t+k+1}
```

where `γ ∈ [0, 1)` is the **discount factor**, controlling the trade-off between immediate and future rewards.

### Key Distinctions from Supervised/Unsupervised Learning

- No labeled dataset: the agent generates its own training data through interaction.
- Reward signal is delayed and sparse: credit assignment is non-trivial.
- The data distribution shifts as the policy improves (non-stationarity).

### Core Components

- **Policy `π`**: a mapping from states to actions (deterministic or stochastic).
- **Value function `V^π(s)`**: expected return from state `s` under policy `π`.
- **Action-value function `Q^π(s, a)`**: expected return from taking action `a` in state `s`, then following `π`.
- **Model** (optional): a learned representation of environment dynamics `P(s' | s, a)`.

---

## 3. Bellman Equation

### State Value Function

The value of a state is defined recursively:

```
V^π(s) = Σ_a π(a|s) · Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V^π(s')]
```

This is the **Bellman Expectation Equation**. It expresses the value of a state as the immediate reward plus the discounted value of the successor state, averaged over both the policy and environment dynamics.

### Bellman Optimality Equation

For the **optimal policy `π*`**:

```
V*(s) = max_a Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V*(s')]
```

```
Q*(s,a) = Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · max_{a'} Q*(s',a')]
```

The optimal Q-function directly encodes the optimal policy: `π*(s) = argmax_a Q*(s,a)`.

### From State Values to Optimal Plans

Value Iteration applies the Bellman optimality equation as an update rule:

```
V_{k+1}(s) ← max_a Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V_k(s')]
```

This is guaranteed to converge to `V*` for finite MDPs under the contraction mapping theorem.

---

## 4. Markov Decision Processes (MDPs)

### Formal Definition

An MDP is a tuple `(S, A, P, R, γ)`:

- `S`: state space
- `A`: action space
- `P(s'|s,a)`: transition probability function
- `R(s,a,s')`: reward function
- `γ`: discount factor

### The Markov Property

The future is conditionally independent of the past given the present:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

This property is what makes the Bellman equations tractable — the full history need not be tracked.

### Types of MDPs

- **Finite horizon**: episode terminates after fixed `T` steps.
- **Infinite horizon discounted**: `γ < 1` ensures convergence.
- **Episodic vs. continuing**: episodic tasks have terminal states; continuing tasks do not.

---

## 5. Optimal Policy vs. Fixed Plans

A **fixed plan** specifies a sequence of actions irrespective of observed states. It is open-loop and brittle — any stochasticity in the environment renders it suboptimal.

A **policy** is a closed-loop mapping: `π: S → A` (or `π: S × A → [0,1]`). It can react to new observations.

The **optimal policy** `π*` satisfies:

```
π*(s) = argmax_a Q*(s, a)  ∀s ∈ S
```

Key insight: any optimal policy derived from `Q*` is Markovian and stationary — no memory beyond the current state is needed when the Markov property holds.

---

## 6. Living Penalty in RL

A **living penalty** (also called step cost or time penalty) is a negative reward applied at every non-terminal timestep:

```
r_t = r_t^{env} - c    (c > 0)
```

### Purpose

- Incentivizes the agent to reach the goal quickly.
- Prevents the agent from idling in states with zero reward.
- Controls the effective discount applied to future rewards.

### Trade-offs

Too large a penalty → agent takes risky shortcuts.  
Too small a penalty → agent meanders, solution time increases.  
Well-calibrated penalty → efficient exploration of the reward landscape.

---

## 7. Q-Learning: From V-Values to Q-Values

### Q-Values

The action-value function `Q(s,a)` is more useful than `V(s)` in model-free settings because it does not require knowledge of `P(s'|s,a)` to extract a policy — the argmax is taken directly over learned Q-values.

### Relationship

```
Q^π(s,a) = Σ_{s'} P(s'|s,a) · [R(s,a,s') + γ · V^π(s')]
V^π(s)   = Σ_a π(a|s) · Q^π(s,a)
```

### Q-Learning (Watkins, 1989)

Q-Learning is a **model-free, off-policy** TD algorithm:

```
Q(s,a) ← Q(s,a) + α · [r + γ · max_{a'} Q(s',a') - Q(s,a)]
```

- **Off-policy**: the target uses `max_{a'}` (greedy), independent of the behavior policy used to generate data.
- **Convergence**: guaranteed to converge to `Q*` under tabular settings given sufficient exploration and decaying learning rate.

---

## 8. Temporal Difference in Q-Learning

### TD Error

The **TD error** (δ) is the core quantity driving Q-Learning updates:

```
δ_t = r_t + γ · max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)
```

This is the difference between the **TD target** (`r + γ · max Q`) and the **current estimate** (`Q(s,a)`).

### TD vs. Monte Carlo

| Property | TD | Monte Carlo |
|---|---|---|
| Update timing | After each step | After episode ends |
| Bias | Biased (bootstraps) | Unbiased |
| Variance | Low | High |
| Applicability | Continuing + episodic | Episodic only |

### Tabular Q-Learning Algorithm

```
Initialize Q(s,a) arbitrarily (e.g., zeros)
For each episode:
  Initialize s
  For each step:
    Choose a using ε-greedy from Q
    Take action a, observe r, s'
    Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]
    s ← s'
  Until s is terminal
```

---

# Part 1 — Deep Q-Learning (DQN)

## 1. Deep Q-Learning vs. Traditional Q-Learning

### The Scalability Problem

Tabular Q-Learning requires storing `|S| × |A|` values. For continuous or high-dimensional state spaces (e.g., pixel observations), this is infeasible.

**Solution**: Replace the Q-table with a neural network `Q(s,a; θ)` parameterized by weights `θ`.

### Key Differences

| Property | Tabular Q-Learning | Deep Q-Learning |
|---|---|---|
| State representation | Discrete, enumerable | Continuous, high-dimensional |
| Q-function storage | Table | Neural network weights |
| Update rule | Direct table update | Gradient descent on loss |
| Convergence guarantees | Proven | Empirical; no formal guarantee |

### The Deadly Triad Problem

Combining function approximation + bootstrapping + off-policy learning introduces instability:

- **Correlated samples**: consecutive transitions are highly correlated, breaking i.i.d. assumptions.
- **Non-stationary targets**: the target `r + γ max Q(s'; θ)` changes as `θ` is updated.

DQN (Mnih et al., 2015) addresses both with **Experience Replay** and a **Target Network**.

---

## 2. How Deep Q-Learning Works

### Architecture

The DQN maps a state `s` to Q-values for all actions simultaneously:

```
Q(s, ·; θ): S → R^|A|
```

This is more efficient than computing Q for each (s,a) pair individually.

### Loss Function

```
L(θ) = E_{(s,a,r,s') ~ D} [ (y - Q(s,a;θ))^2 ]
y = r + γ · max_{a'} Q(s', a'; θ^-)
```

where `θ^-` are the **target network** weights (a frozen copy of `θ`).

### Target Network

- A copy of the Q-network updated periodically (every `C` steps).
- Decouples the target from the online network, stabilizing training.
- **Hard update**: `θ^- ← θ` every `C` steps.
- **Soft update**: `θ^- ← τθ + (1-τ)θ^-` at every step (used in later variants).

---

## 3. Experience Replay

### Motivation

Raw online RL data violates the i.i.d. assumption required by stochastic gradient descent:

- Sequential transitions are temporally correlated.
- The agent's current policy determines the data distribution, creating feedback loops.

### Replay Buffer

A fixed-size circular buffer `D` storing tuples `(s, a, r, s', done)`.

```
D = deque(maxlen=N)
```

At each training step:
1. Execute action, store transition in `D`.
2. Sample a random mini-batch from `D`.
3. Compute TD targets and update `θ` via gradient descent.

### Benefits

- Breaks temporal correlations by random sampling.
- Allows each transition to be replayed multiple times (sample efficiency).
- Decouples data collection rate from network update rate.

### Hyperparameters

- **Buffer size `N`**: typically 10,000–1,000,000.
- **Batch size**: typically 32–256.
- **Minimum fill**: begin training only after buffer has `≥ batch_size` transitions.

---

## 4. Action Selection: ε-Greedy and Softmax

### Exploration-Exploitation Dilemma

The agent must balance:
- **Exploitation**: select the action believed to be optimal (maximize immediate Q).
- **Exploration**: try suboptimal actions to discover better long-term strategies.

### ε-Greedy Policy

```
a = argmax_a Q(s,a)    with probability 1-ε
a ~ Uniform(A)          with probability ε
```

**ε-decay schedule** (common configuration):

```
ε_t = max(ε_min, ε_start · decay^t)
```

Typical values: `ε_start = 1.0`, `ε_min = 0.01`, `decay = 0.995`.

### Softmax (Boltzmann) Policy

```
π(a|s) = exp(Q(s,a) / τ) / Σ_{a'} exp(Q(s,a') / τ)
```

`τ` is the **temperature** parameter:
- High `τ` → near-uniform (maximum exploration).
- Low `τ` → near-greedy (maximum exploitation).

### Comparison

| Property | ε-Greedy | Softmax |
|---|---|---|
| Relative Q magnitudes used | No | Yes |
| Simplicity | High | Medium |
| Sensitive to Q scale | No | Yes |

---

## 5. Lunar Lander Implementation (PyTorch DQN)

### Environment

`LunarLander-v2` from Gymnasium:
- **State**: 8-dimensional continuous vector (position, velocity, angle, angular velocity, leg contact).
- **Actions**: 4 discrete (do nothing, fire left, fire main, fire right).
- **Reward**: shaped; episode terminates on crash or landing.
- **Solved**: average score ≥ 200 over 100 episodes.

### Network Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # raw Q-values; no activation
```

### Replay Memory

```python
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```

### DQN Agent

```python
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99,
                 buffer_size=10000, batch_size=64, tau=1e-3):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayMemory(buffer_size)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q_network(state_t).argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self.learn()

    def learn(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states      = torch.FloatTensor(batch.state)
        actions     = torch.LongTensor(batch.action).unsqueeze(1)
        rewards     = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_states = torch.FloatTensor(batch.next_state)
        dones       = torch.FloatTensor(batch.done).unsqueeze(1)

        q_values  = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update()

    def _soft_update(self):
        for target_p, online_p in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_p.data.copy_(self.tau * online_p.data + (1 - self.tau) * target_p.data)
```

### Training Loop

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2")
agent = DQNAgent(state_dim=8, action_dim=4)

n_episodes = 2000
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.01, 0.995
epsilon = epsilon_start
scores, scores_window = [], deque(maxlen=100)

for ep in range(n_episodes):
    state, _ = env.reset()
    score = 0
    done = False

    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, float(done))
        state = next_state
        score += reward

    epsilon = max(epsilon_end, epsilon * epsilon_decay)
    scores.append(score)
    scores_window.append(score)

    if ep % 100 == 0:
        print(f"Episode {ep}\tAverage Score: {np.mean(scores_window):.2f}")
    if np.mean(scores_window) >= 200.0:
        print(f"Environment solved in {ep} episodes!")
        break
```

---

# Part 2 — Deep Convolutional Q-Learning (DCQN)

## 1. Why Convolutional Networks for RL?

Standard fully-connected DQNs require a structured state vector. When the agent's observation is raw pixel data (e.g., Atari games), a CNN is required to:

- Extract spatial features from 2D/3D input.
- Learn translation-invariant representations.
- Reduce input dimensionality before value estimation.

## 2. Deep Convolutional Q-Learning vs. DQN

| Component | DQN | DCQN |
|---|---|---|
| Input | Feature vector | Raw pixels (frames) |
| Feature extractor | FC layers | Convolutional layers |
| Frame stacking | N/A | Yes (typically 4 frames) |
| Preprocessing | Normalization | Grayscale + resize + normalize |
| Memory footprint | Lower | Higher |

## 3. Eligibility Traces vs. DCQN

**Eligibility traces** (λ-returns) propagate credit backwards through time using a decay factor `λ`:

```
G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t^{(n)}
```

- `λ = 0`: pure TD(0) — single-step bootstrapping.
- `λ = 1`: Monte Carlo — full episode return.

DCQN uses single-step TD (`λ = 0`) but compensates with large replay buffers and convolutional feature extraction. Eligibility traces are rarely used with neural function approximators due to instability.

## 4. CNN Architecture for Atari (Ms. Pac-Man)

### Preprocessing Pipeline

```python
from PIL import Image
import torchvision.transforms as T
import torch

def preprocess_frame(frame):
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        T.Resize((84, 84)),
        T.ToTensor(),            # scales to [0, 1]
    ])
    return transform(frame)     # shape: (1, 84, 84)
```

Frame stacking: concatenate last `k=4` preprocessed frames → input shape `(4, 84, 84)`.

### Convolutional Architecture

```python
class DCQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # Input: (batch, 4, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)   # flatten
        return self.fc(x)
```

This is the original DQN architecture from Mnih et al. (2015), adapted for the Pac-Man action space.

### Convolution Output Size Formula

```
output_size = floor((input_size - kernel_size) / stride) + 1
```

### Training on GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DCQN(action_dim=9).to(device)

# States must be moved to GPU before forward pass
states = torch.FloatTensor(batch.state).to(device)
```

V100 GPU significantly reduces per-epoch training time for pixel-based environments — this is practically mandatory for Atari-scale training.

---

# Part 3 — Asynchronous Advantage Actor-Critic (A3C)

## 1. From DQN to Actor-Critic

### Limitations of DQN

- Requires large replay buffers (memory intensive).
- Training is sequential (one process, one environment).
- Handles only discrete action spaces naturally.
- Experience replay breaks temporal correlations — but at a storage cost.

### Actor-Critic Architecture

Instead of learning only `Q(s,a)`, Actor-Critic methods maintain two separate components:

- **Actor**: a policy `π(a|s; θ_π)` that selects actions.
- **Critic**: a value function `V(s; θ_v)` that evaluates states.

The critic provides a **baseline** to reduce variance in policy gradient updates.

### Policy Gradient Theorem

The gradient of the expected return with respect to `θ_π` is:

```
∇_{θ_π} J(θ_π) = E_π [ ∇_{θ_π} log π(a|s; θ_π) · Q^π(s,a) ]
```

Subtracting a baseline `b(s)` (e.g., `V(s)`) reduces variance without introducing bias:

```
∇_{θ_π} J = E_π [ ∇ log π(a|s) · (Q(s,a) - V(s)) ]
           = E_π [ ∇ log π(a|s) · A(s,a) ]
```

where `A(s,a) = Q(s,a) - V(s)` is the **advantage function**.

---

## 2. Advantage Function

The advantage `A(s,a)` quantifies how much better action `a` is compared to the average action in state `s`:

- `A(s,a) > 0`: action `a` is better than average → increase its probability.
- `A(s,a) < 0`: action `a` is worse than average → decrease its probability.
- `A(s,a) = 0`: action `a` is average.

### TD Advantage Estimate

In practice, `Q(s,a)` is not known. We estimate it using a single TD step:

```
A(s_t, a_t) ≈ r_t + γ · V(s_{t+1}; θ_v) - V(s_t; θ_v)
```

This is the **TD error** used as the advantage estimate — unbiased given accurate `V`, but biased otherwise.

### n-Step Advantage

For better bias-variance trade-off:

```
A_t^{(n)} = Σ_{k=0}^{n-1} γ^k · r_{t+k} + γ^n · V(s_{t+n}) - V(s_t)
```

---

## 3. A3C: Asynchronous Advantage Actor-Critic

### Key Innovation (Mnih et al., 2016)

A3C eliminates the replay buffer entirely by running **multiple agents in parallel** with separate environment instances. Asynchronous gradient updates to a **shared global network** decorrelate the training data.

### Architecture

```
Global Network (shared θ_π, θ_v)
    │
    ├── Worker 1 (env_1) → local gradients → async update global
    ├── Worker 2 (env_2) → local gradients → async update global
    ├── Worker 3 (env_3) → local gradients → async update global
    └── ... (typically 8-16 workers)
```

Each worker:
1. Copies global weights to its local network.
2. Interacts with its environment for `t_max` steps.
3. Computes gradients for actor and critic losses.
4. Applies gradients to the **global** network (asynchronously, without locks).
5. Repeats.

### Shared Critic

Workers share the same global critic — each worker's environment provides a different trajectory, improving coverage of the state space without a replay buffer.

---

## 4. LSTM in A3C

### Motivation

Many environments are **partially observable** — the current frame alone does not contain sufficient information. LSTM layers provide the agent with memory of past observations.

### LSTM Recap

The LSTM maintains a hidden state `(h_t, c_t)`:

```
i_t = σ(W_i x_t + U_i h_{t-1} + b_i)   # input gate
f_t = σ(W_f x_t + U_f h_{t-1} + b_f)   # forget gate
o_t = σ(W_o x_t + U_o h_{t-1} + b_o)   # output gate
g_t = tanh(W_g x_t + U_g h_{t-1} + b_g) # cell input
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

In A3C, the LSTM receives the CNN feature vector and outputs a representation used by both the actor and critic heads.

---

## 5. A3C Implementation: KungFu Master (Atari)

### Network Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class A3CNetwork(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # Visual feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  nn.ELU(),
        )
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        # Actor and Critic heads
        self.actor  = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x, hx, cx):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        return F.softmax(self.actor(hx), dim=-1), self.critic(hx), hx, cx
```

### Atari Preprocessing

```python
class PreprocessAtari:
    def __init__(self, env, frame_skip=4, new_size=(42, 42)):
        self.env = env
        self.frame_skip = frame_skip
        self.new_size = new_size

    def process(self, frame):
        frame = Image.fromarray(frame).convert('L')   # grayscale
        frame = frame.resize(self.new_size)
        return np.array(frame) / 255.0               # normalize

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done: break
        return self.process(obs), total_reward, done or trunc, info
```

### Loss Functions

**Critic loss** (value function regression):

```python
value_loss = F.mse_loss(values, returns)
```

**Actor loss** (policy gradient with entropy regularization):

```python
log_probs   = torch.log(probs + 1e-8)
entropy     = -(probs * log_probs).sum(dim=-1).mean()
policy_loss = -(log_probs_taken * advantages.detach()).mean() - 0.01 * entropy
```

The entropy bonus discourages premature convergence to a deterministic policy.

**Total loss**:

```python
loss = policy_loss + 0.5 * value_loss
```

### EnvBatch for Multi-Environment Training

```python
class EnvBatch:
    def __init__(self, env_name, n_envs=10):
        self.envs = [gym.make(env_name) for _ in range(n_envs)]

    def reset(self):
        return [env.reset()[0] for env in self.envs]

    def step(self, actions):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rews, dones, truncs, infos = zip(*results)
        return list(obs), list(rews), list(dones), list(truncs), list(infos)
```

### Training Loop Overview

```python
# Pseudocode for A3C training loop
global_network = A3CNetwork(action_dim)
optimizer = torch.optim.Adam(global_network.parameters(), lr=1e-4)

for step in range(total_steps):
    # Sync local network from global
    local_network.load_state_dict(global_network.state_dict())

    # Collect n-step trajectory
    states, actions, rewards, values, log_probs = [], [], [], [], []
    for _ in range(n_steps):
        probs, value, hx, cx = local_network(state, hx, cx)
        action = Categorical(probs).sample()
        next_state, reward, done, _ = env.step(action.item())
        # store...
        state = next_state

    # Compute returns and advantages
    returns   = compute_returns(rewards, values, gamma)
    advantage = returns - torch.stack(values)

    # Compute losses and update global
    loss = actor_loss + 0.5 * critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

# Part 4 — AWS DeepRacer (PPO and SAC)

## Overview

AWS DeepRacer is a 1/18th-scale autonomous vehicle trained via RL in a simulated track environment. The system uses:

- **PPO (Proximal Policy Optimization)**: on-policy, clipped surrogate objective.
- **SAC (Soft Actor-Critic)**: off-policy, maximum entropy framework.

### PPO

PPO constrains policy updates to prevent large deviations from the current policy:

```
L^{CLIP}(θ) = E_t [ min(r_t(θ) · A_t, clip(r_t(θ), 1-ε, 1+ε) · A_t) ]
```

where `r_t(θ) = π_θ(a|s) / π_{θ_old}(a|s)` is the probability ratio.

- `ε = 0.2` is a common default.
- Conservative updates → more stable than TRPO but without second-order computations.

### SAC

SAC maximizes a modified objective that includes an entropy term:

```
J(π) = Σ_t E [ r(s_t, a_t) + α · H(π(·|s_t)) ]
```

- `α` (temperature): controls the entropy-reward trade-off.
- Dual Q-networks to mitigate overestimation bias.
- Off-policy → more sample efficient than PPO.

### DeepRacer Reward Function Design

The reward function is user-defined in Python. Key signals:

- `distance_from_center`: penalize deviation from track center.
- `steering_angle`: penalize sharp turns to encourage smooth trajectories.
- `speed`: reward higher speeds when on track.
- `all_wheels_on_track`: binary flag; penalize off-track.

---

# Part 5 — Introduction to Large Language Models (LLMs)

## 1. What is an LLM?

A **Large Language Model** is a transformer-based neural network trained on massive text corpora to model the probability distribution over token sequences:

```
P(x_1, x_2, ..., x_n) = Π_{i=1}^{n} P(x_i | x_1, ..., x_{i-1})
```

The model learns to predict the next token given all preceding tokens — **autoregressive language modeling**.

## 2. Transformer Architecture

### Attention Mechanism

Self-attention computes queries, keys, and values from the input:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

- `Q = XW_Q`, `K = XW_K`, `V = XW_V`
- `d_k`: dimension of key vectors (scaling prevents vanishing gradients in softmax)

**Multi-Head Attention** runs `h` attention heads in parallel, concatenating results:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W_O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### Transformer Block

```
x → LayerNorm → Multi-Head Self-Attention → Residual Add
  → LayerNorm → Feed-Forward Network → Residual Add
```

FFN: `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`

### Positional Encoding

Since attention is permutation-invariant, positional information must be injected:

```
PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
```

Modern LLMs use **Rotary Position Embeddings (RoPE)** instead.

## 3. Essential Ingredients for LLM Development

1. **Training data**: diverse, large-scale text corpus (Common Crawl, Books, Wikipedia, code).
2. **Tokenizer**: BPE (Byte-Pair Encoding) or SentencePiece to convert text to token IDs.
3. **Model architecture**: transformer with causal masking for autoregressive generation.
4. **Compute**: thousands of GPU/TPU hours at scale.
5. **Training objective**: cross-entropy loss over next-token prediction.
6. **Alignment phase**: RLHF (Reinforcement Learning from Human Feedback) or DPO.

## 4. Origins of the Transformer

Key milestones:

- **2017**: "Attention Is All You Need" (Vaswani et al.) — original Transformer.
- **2018**: GPT-1 (OpenAI) — decoder-only autoregressive LM.
- **2018**: BERT (Google) — encoder-only, masked LM.
- **2020**: GPT-3 — 175B parameters, few-shot learning emerges.
- **2023**: LLaMA (Meta) — open-weight, efficient training.

## 5. Next-Token Prediction

At inference time, the model generates text autoregressively:

```
for each generation step:
    logits = model(token_ids)          # shape: (seq_len, vocab_size)
    next_token_logits = logits[-1]     # last position
    probs = softmax(next_token_logits / temperature)
    next_token = sample(probs)         # or argmax for greedy
    token_ids.append(next_token)
```

**Sampling strategies**:
- **Greedy**: `argmax(logits)` — deterministic, often repetitive.
- **Top-k**: sample from top-k highest probability tokens.
- **Top-p (nucleus)**: sample from smallest set of tokens whose cumulative probability exceeds `p`.
- **Temperature scaling**: dividing logits by `T < 1` sharpens distribution; `T > 1` flattens it.

## 6. LLM Parameters

Model size is measured in parameter count (`|θ|`). Parameters are located in:

- Embedding matrices.
- Weight matrices in self-attention (`W_Q, W_K, W_V, W_O`).
- Feed-forward layer weights.
- Layer normalization scales and biases.

Scaling laws (Chinchilla, Hoffmann et al., 2022) show that **optimal training** requires roughly 20 tokens per parameter.

## 7. Context Window

The context window defines the maximum sequence length the model can attend to. It is bounded by the quadratic complexity of full attention: `O(n^2 · d)`.

Modern techniques to extend context:
- **Sliding window attention** (Longformer).
- **Flash Attention**: IO-aware exact attention, memory efficient.
- **RoPE with NTK scaling**: extends positional encoding range.

## 8. Fine-Tuning LLMs

### Full Fine-Tuning

Update all `|θ|` parameters on a domain-specific dataset. Expensive; risks catastrophic forgetting.

### Parameter-Efficient Fine-Tuning (PEFT)

Update only a small fraction of parameters:

**LoRA (Low-Rank Adaptation)** — Hu et al., 2021:

```
W' = W + ΔW = W + B · A
```

Where `B ∈ R^{d×r}` and `A ∈ R^{r×k}` with rank `r << min(d,k)`. Only `A` and `B` are trained.

**QLoRA** — combines 4-bit quantization with LoRA to fine-tune 65B+ models on a single GPU.

---

## 9. Fine-Tuning LLaMA 2 for a Medical Chatbot

### Setup and Dependencies

```python
# Key libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import bitsandbytes as bnb
import torch
```

### 4-bit Quantization with BitsAndBytes

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4: optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,     # nested quantization for further memory savings
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
```

### LoRA Configuration

```python
lora_config = LoraConfig(
    r=16,               # rank
    lora_alpha=32,      # scaling factor (effectively lr multiplier)
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # typically ~0.1% of total
```

### Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"    # required for causal LM
```

### Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./llama2-medical",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    report_to="none",
)
```

### SFTTrainer

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
```

### Inference

```python
def chat(prompt, model, tokenizer, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

# Appendix — Deep Learning Fundamentals

## A. What is Deep Learning?

Deep Learning is the subfield of Machine Learning that uses multi-layer neural networks to learn hierarchical representations directly from raw data. The key insight is that each layer learns increasingly abstract features:

```
Input → edges → textures → parts → objects (for image classification)
```

## B. Artificial Neurons

A single neuron computes a weighted sum followed by a nonlinearity:

```
y = f(Σ_i w_i · x_i + b)
```

Multiple neurons in a layer share the same input but have independent weights.

## C. Backpropagation

Backpropagation applies the **chain rule** to compute gradients of the loss with respect to all parameters:

```
∂L/∂w_i = ∂L/∂y · ∂y/∂z · ∂z/∂w_i
```

where `z = Σ w_i x_i + b` and `y = f(z)`.

### Gradient Descent Variants

| Variant | Update per step | Notes |
|---|---|---|
| Batch GD | Full dataset | Stable; slow per update |
| Stochastic GD | Single sample | Noisy; fast per update |
| Mini-batch GD | Small subset (32-256) | Best of both; standard |

### Adam Optimizer

Combines momentum (first moment) and RMSProp (second moment):

```
m_t = β_1 m_{t-1} + (1-β_1) g_t
v_t = β_2 v_{t-1} + (1-β_2) g_t^2
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

Default: `β_1 = 0.9`, `β_2 = 0.999`, `ε = 1e-8`.

## D. Convolutional Neural Networks (CNNs)

### Convolution Operation

```
(I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] · K[m,n]
```

- `K`: learnable kernel/filter.
- Output: **feature map** highlighting patterns matching `K`.
- Multiple filters → multiple feature maps.

### Max Pooling

Reduces spatial dimensions by taking the maximum in each window:

```
output[i,j] = max_{m,n ∈ window} input[i·s+m, j·s+n]
```

- Provides translational invariance.
- Reduces memory and computation.

### Flattening and Fully Connected Layers

After convolutional layers, the feature maps are flattened into a 1D vector and passed through FC layers for classification or regression.

### Softmax and Cross-Entropy

For classification with `C` classes:

**Softmax**: `p_i = e^{z_i} / Σ_j e^{z_j}`

**Cross-Entropy loss**: `L = -Σ_i y_i · log(p_i)`

where `y` is the one-hot ground truth vector.

The gradient of cross-entropy after softmax simplifies elegantly:

```
∂L/∂z_i = p_i - y_i
```

---

*End of course notes.*

---

# Supplement — Missing Topics (Extra AIs + NLP for LLMs)

## Extra AI 1 — DDPG: Deep Deterministic Policy Gradient

### Motivation

DQN and its convolutional variants handle only **discrete** action spaces. For continuous control (robotics, autonomous driving, continuous locomotion), taking the `argmax` over actions is intractable. DDPG (Lillicrap et al., 2015) resolves this by extending Deep Q-Learning to continuous action domains.

### Key Properties

- **Model-free**, **off-policy**, **actor-critic**.
- Learns a **deterministic** policy `μ(s; θ^μ): S → A` rather than a stochastic distribution.
- Maintains a **replay buffer** (inherited from DQN) to break temporal correlations.
- Uses **target networks** for both actor and critic (soft/Polyak updates).

### Why Deterministic Policy?

A stochastic policy `π(a|s)` requires integrating over the action space to compute the policy gradient — intractable in continuous spaces. The **Deterministic Policy Gradient Theorem** (Silver et al., 2014) shows:

```
∇_{θ^μ} J = E_s [ ∇_a Q(s,a; θ^Q)|_{a=μ(s)} · ∇_{θ^μ} μ(s; θ^μ) ]
```

The gradient of Q with respect to action is well-defined and backpropagable — no integration required. This is the key mathematical insight enabling DDPG.

### Architecture

```
Four networks (all neural networks):
  1. Online Actor:   μ(s; θ^μ)          — outputs deterministic action
  2. Online Critic:  Q(s,a; θ^Q)        — evaluates (state, action) pairs
  3. Target Actor:   μ'(s; θ^μ')        — frozen copy of actor
  4. Target Critic:  Q'(s,a; θ^Q')      — frozen copy of critic
```

### Training Procedure

**Critic update** — minimize the Bellman error:

```
y_i = r_i + γ · Q'(s_{i+1}, μ'(s_{i+1}; θ^μ'); θ^Q')
L(θ^Q) = (1/N) Σ_i (y_i - Q(s_i, a_i; θ^Q))^2
```

**Actor update** — gradient ascent on Q via the deterministic policy gradient:

```
∇_{θ^μ} J ≈ (1/N) Σ_i ∇_a Q(s,a; θ^Q)|_{a=μ(s_i)} · ∇_{θ^μ} μ(s_i; θ^μ)
```

**Target network soft update** (Polyak averaging, applied after every step):

```
θ^Q'  ← τ θ^Q  + (1-τ) θ^Q'
θ^μ'  ← τ θ^μ  + (1-τ) θ^μ'      (τ ≈ 0.005)
```

### Exploration via Noise Injection

Since the policy is deterministic, exploration must be explicitly injected. During training:

```
a_t = μ(s_t; θ^μ) + ε_t
```

Two noise options:
- **Ornstein-Uhlenbeck (OU) process**: temporally correlated noise (original paper). Mean-reverting: `dε = θ(μ-ε)dt + σ dW`. Useful for inertial physical systems.
- **Gaussian noise** `ε ~ N(0, σ²)`: uncorrelated. Simpler; empirically equivalent or better.

At test time, noise is removed: `a = μ(s)`.

### PyTorch Implementation Sketch

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
            nn.Linear(256, action_dim),nn.Tanh(),  # bounded output
        )
        self.max_action = max_action

    def forward(self, s):
        return self.max_action * self.net(s)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),                    nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))
```

### DDPG vs DQN vs A3C

| Property | DQN | A3C | DDPG |
|---|---|---|---|
| Action space | Discrete | Discrete | Continuous |
| Policy type | Implicit (argmax Q) | Stochastic | Deterministic |
| Off-policy | Yes | No | Yes |
| Replay buffer | Yes | No | Yes |
| Target network | Yes | No | Yes (both actor+critic) |
| Parallelism | No | Yes | No |

---

## Extra AI 2 — Full World Model (Ha & Schmidhuber, 2018)

### Concept

A **world model** is an internal generative model of the environment that allows an agent to simulate future states without interacting with the real environment. The agent can learn and plan entirely inside its own "dream."

Reference: Ha & Schmidhuber, "World Models" (NeurIPS 2018, arXiv:1803.10122).

### Three-Component Architecture

```
Observation (x_t)
    │
    ▼
[V] Vision: VAE
    Encodes x_t → latent z_t (compact spatial representation)
    │
    ▼
[M] Memory: MDN-RNN
    Given (z_t, a_t), predicts distribution over z_{t+1}
    Outputs hidden state h_t
    │
    ▼
[C] Controller: Linear policy
    Takes (z_t, h_t) as input → outputs action a_t
    Trained by CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
```

### V — Vision (Variational Autoencoder)

The VAE compresses high-dimensional pixel observations into a low-dimensional latent vector `z`:

```
Encoder: q_φ(z|x) = N(μ_φ(x), σ_φ(x))
Decoder: p_θ(x|z)
Loss: L = E[log p_θ(x|z)] - KL(q_φ(z|x) || N(0,I))
```

- The KL term regularizes the latent space to be smooth and continuous.
- Only `z` (not the reconstruction) is passed to M and C.

### M — Memory (MDN-RNN)

The memory component models environment dynamics in latent space using a **Mixture Density Network** on top of an **LSTM**:

```
h_t, c_t = LSTM(z_t, a_t, h_{t-1}, c_{t-1})
P(z_{t+1} | h_t) = ΣK π_k · N(μ_k, σ_k)   (mixture of Gaussians)
```

This predicts a distribution over next latent states, capturing uncertainty. The hidden state `h_t` summarizes temporal context.

### C — Controller (Linear Policy)

The controller is deliberately **minimal** — a single linear layer:

```
a_t = W_c [z_t ; h_t] + b_c
```

This design keeps the search space small for the evolutionary optimizer. `[z_t ; h_t]` is the concatenation of current latent observation and RNN hidden state.

### Training Pipeline

1. **Collect rollouts**: run a random policy in the real environment; store `(x_t, a_t, r_t)` tuples.
2. **Train V**: fit the VAE on all observations `{x_t}` to learn the encoder/decoder.
3. **Encode and train M**: encode all observations to `{z_t}`; train MDN-RNN on sequences `(z_t, a_t) → z_{t+1}`.
4. **Train C in dream**: run the controller entirely inside M's simulated environment (no real environment interaction); optimize C's weights using **CMA-ES**.
5. **Transfer**: deploy the policy learned in the dream back to the real environment.

### Dream Training and Temperature

When training inside the dream, the MDN-RNN samples next states from its predicted distribution:

```
z_{t+1} ~ P(z_{t+1} | h_t; τ)
```

**Temperature parameter `τ`** controls uncertainty:
- Low `τ` (≈0.1): near-deterministic dream → agent exploits model artifacts, fails in reality.
- Optimal `τ` (≈1.15): sufficient stochasticity → policy generalizes to the real environment.
- High `τ`: too chaotic to learn.

### CMA-ES for Controller Optimization

CMA-ES is an evolutionary strategy that maintains a multivariate Gaussian over policy parameters:

```
θ_i ~ N(μ, Σ)      (population of candidate policies)
```

At each generation:
1. Sample population `{θ_1, ..., θ_N}`.
2. Evaluate each in the dream (cumulative reward).
3. Update `μ` and `Σ` toward high-reward parameter regions.

No backpropagation is required through the controller. This makes training robust to sparse rewards and long horizons.

### Strengths and Limitations

**Strengths:**
- Agent can train in a dream — dramatically reduces real-world sample requirements.
- Modular: V, M, C can be trained independently.
- M's uncertainty can serve as an intrinsic exploration signal.

**Limitations:**
- Catastrophic forgetting in LSTM-based M with limited capacity.
- No hierarchical planning — step-by-step simulation only.
- Exploitability: adversarial policies can find model artifacts (partially mitigated by temperature).

---

## Extra AI 3 — Evolution Strategies & Genetic Algorithms

### Genetic Algorithms (GA)

Genetic Algorithms are population-based optimization methods inspired (loosely) by biological evolution. They operate on a **population of candidate solutions** and apply evolutionary operators:

```
Population: {θ_1, θ_2, ..., θ_N}

Cycle:
  1. Selection: select top-k by fitness f(θ_i)
  2. Crossover: combine pairs of parents → offspring
  3. Mutation: perturb offspring with noise
  4. Replace: new generation = offspring + elites
```

**Fitness function**: in RL, `f(θ) = E[Σ r_t]` — the cumulative reward under policy `θ`.

**No gradients required**: fitness is evaluated as a black-box scalar. This makes GAs applicable where reward is non-differentiable, sparse, or delayed.

### Evolution Strategies (ES)

Evolution Strategies are a specific family of evolutionary algorithms focused on **continuous parameter optimization**. The OpenAI ES formulation (Salimans et al., 2017):

**Setup**: parameterize the search distribution as `θ ~ N(θ̄, σ²I)`.

**Update rule**:

```
θ_i = θ̄ + σ · ε_i,    ε_i ~ N(0, I)     (sample population)
F_i = f(θ_i)                               (evaluate fitness)
θ̄ ← θ̄ + α/(N·σ) · Σ_i F_i · ε_i        (weighted update)
```

This is mathematically equivalent to REINFORCE applied to the policy over parameter vectors. The key innovation: **only scalar fitness scores are communicated between workers** — no gradients, no backpropagation.

### ES vs. Standard RL

| Property | Gradient-based RL (PPO/A3C) | Evolution Strategies |
|---|---|---|
| Gradient source | Backprop through rewards | Black-box finite differences |
| Parallelism | Limited (correlated updates) | Embarrassingly parallel (linear speedup) |
| Sparse rewards | Problematic | Naturally handled |
| Delayed rewards | Problematic (credit assignment) | Invariant to time horizon |
| Memory per worker | Full model + gradients | Only random seeds + scalar |
| Communication cost | High (gradients) | Minimal (scalars) |
| Sample efficiency | Higher | Lower (3–10× more samples) |

### Scaling Property

At each step, ES generates a population of candidate parameter vectors by adding Gaussian noise, evaluates each independently, then forms the updated parameters as a weighted sum proportional to each candidate's total reward. Because workers only exchange seeds and scalar scores, using 720 cores, ES can achieve performance comparable to A3C on Atari while reducing training time from roughly one day to one hour.

### Mirror Sampling

To reduce gradient estimate variance, ES uses **antithetic sampling** (mirror sampling):

```
θ_i^+ = θ̄ + σ · ε_i
θ_i^- = θ̄ - σ · ε_i

gradient ≈ (1/2Nσ) Σ_i (F(θ_i^+) - F(θ_i^-)) · ε_i
```

This is equivalent to a control variate, halving the variance at no additional cost.

### Fitness Shaping (Rank Normalization)

Raw rewards can have high variance and outliers. Fitness shaping applies rank transformation:

```
F̃_i = rank(F_i) / N - 0.5
```

This makes the update invariant to monotonic transformations of the reward function and prevents reward outliers from dominating.

### When to Use ES vs. Gradient-based RL

- Use **gradient-based RL** (PPO, SAC, A3C) when: sample efficiency matters, the environment allows rapid interaction, rewards are dense.
- Use **ES** when: massive parallelism is available, rewards are sparse or delayed, backpropagation is infeasible (e.g., non-differentiable simulators), or the policy has very few parameters (controller in World Models).

---

## NLP Techniques for LLMs: Tokenization and Padding

### Tokenization

Tokenization converts raw text strings into sequences of integer token IDs that the model processes. The vocabulary `V` is fixed at training time.

#### Byte-Pair Encoding (BPE)

BPE is the standard tokenizer for GPT-family models (including LLaMA):

**Training algorithm**:
1. Initialize vocabulary with individual characters.
2. Count all adjacent symbol pairs in the corpus.
3. Merge the most frequent pair into a new symbol.
4. Repeat until `|V|` reaches target size.

**Result**: frequent subwords get single tokens; rare words split into known subword pieces.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokens = tokenizer("Hello, how are you?")
# {'input_ids': [1, 15043, 29892, 920, 526, 366, 29973], ...}
```

Key properties:
- Out-of-vocabulary words never occur (byte fallback handles any Unicode).
- Medical/domain terminology may be split into many tokens, increasing sequence length and reducing attention efficiency.

#### SentencePiece

Used by models like LLaMA and T5. Trains directly on raw text without pre-tokenization, handling languages with no whitespace boundaries (Japanese, Chinese).

### Padding and Truncation

LLMs require fixed-length input tensors within a batch. Padding aligns variable-length sequences:

```
Sequence 1: [101, 2023, 2003, 1037, 3231, 102, PAD, PAD]
Sequence 2: [101, 7592, 102, PAD, PAD, PAD, PAD, PAD]
Sequence 3: [101, 1045, 2293, 17953, 102, PAD, PAD, PAD]
```

**Attention mask**: a binary tensor marking real tokens (1) vs. padding (0):

```python
encoding = tokenizer(
    texts,
    padding="longest",          # pad to longest in batch
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
# encoding.input_ids:      (batch, seq_len)
# encoding.attention_mask: (batch, seq_len) — 0 for PAD tokens
```

**Padding side**:
- `padding_side = "right"`: required for causal LMs during training (prevents attending to future tokens after padding).
- `padding_side = "left"`: used at inference time for batch generation with decoder-only models.

### Knowledge Augmentation in Fine-Tuning

Standard fine-tuning adapts the model's parameters to a target domain but does not inject new factual knowledge beyond what was encoded in the base model's weights. **Knowledge augmentation** refers to augmenting training samples with retrieved or structured external knowledge.

Techniques:

1. **Retrieval-Augmented Generation (RAG)**: at inference time, retrieve relevant documents from a knowledge base and prepend them to the prompt:
   ```
   [Retrieved context] + [User query] → LLM → [Answer]
   ```
   The model doesn't need to memorize facts — it reads them from context.

2. **Knowledge-Augmented Fine-Tuning**: during SFT, each training sample includes retrieved context:
   ```
   {"text": "[Document: ...relevant text...]\nQuestion: ...\nAnswer: ..."}
   ```
   This teaches the model to reason over retrieved context rather than relying solely on weights.

3. **Structured knowledge injection**: format domain ontologies, medical code systems (ICD-10, SNOMED), or drug databases as text and include them in instruction-tuning data.

In the **medical chatbot** context (LLaMA-2 fine-tuning), this means including drug interaction tables, clinical guidelines, or symptom-diagnosis mappings as part of the training set, so the model learns to correctly use structured medical information in its responses.

---

## Q-Learning Case Study: Process Optimization (Warehouse Flow)

### Problem Formulation

A **warehouse flow optimization** problem maps naturally to an MDP:

- **State** `s`: current location of a robot/agent within the warehouse graph.
- **Action** `a`: move to an adjacent location (node in the graph).
- **Reward** `R(s,a)`: negative unless the agent reaches high-priority delivery zones or the goal.
- **Terminal state**: the agent reaches the dispatch point (goal location).

The warehouse is represented as a directed graph where edges have associated rewards. Priority zones yield positive rewards; all other transitions incur the living penalty.

### Q-Table Initialization and Training

```python
import numpy as np

# State: 12 warehouse locations (nodes 0-11)
# Actions: transitions defined by the graph adjacency
n_states  = 12
n_actions = 12

# Reward matrix R[s,a]: -1 for invalid/low-priority, ≥0 for valid, 100 for goal
R = np.full((n_states, n_actions), -1)
# (fill with domain-specific rewards)

Q = np.zeros((n_states, n_actions))  # initialize Q-table to zeros

alpha, gamma = 0.9, 0.75
n_episodes = 1000

for _ in range(n_episodes):
    s = np.random.randint(0, n_states)        # random start state
    while s != goal_state:
        valid_actions = [a for a in range(n_actions) if R[s,a] >= 0]
        a = np.random.choice(valid_actions)   # random exploration (ε=1)
        s_next = a                             # in this graph, action = next state
        Q[s,a] += alpha * (R[s,a] + gamma * Q[s_next].max() - Q[s,a])
        s = s_next

# Optimal route from state s0:
def optimal_route(start):
    route, s = [start], start
    while s != goal_state:
        a = Q[s].argmax()
        route.append(a)
        s = a
    return route
```

This tabular approach works because the state/action space is finite and small. The Q-table converges to encode the globally optimal path from any starting location to the goal.

