# Ex-6: Implementation-of-MC-prediction-for-estimating-the-action-value-function.

## Aim

To implement the Monte Carlo (MC) Prediction algorithm for estimating the action-value function \(Q(s,a)\) using sampled episodes and to analyze the learned action values in a Grid World environment.

---

## Objective

- To understand Monte Carlo Prediction for action-value estimation.
- To estimate the action-value function \(Q(s,a)\).
- To learn state-action values from complete episodes.
- To evaluate the quality of actions under a given policy.

---

## Theory

Monte Carlo Prediction is a model-free reinforcement learning technique used to estimate value functions directly from experience.

The action-value function is defined as:

\[
Q(s,a) = E[G_t \mid S_t=s, A_t=a]
\]

Where:

- \(Q(s,a)\) = Expected return for taking action \(a\) in state \(s\)
- \(G_t\) = Discounted return after time step \(t\)

Monte Carlo methods estimate action values by averaging returns obtained after visiting each state-action pair over many episodes.

---

## Algorithm

### Monte Carlo Prediction for Action-Value Function

1. Initialize:
   - Action-value function \(Q(s,a)\)
   - Returns list for every state-action pair

2. Generate an episode using a policy.

3. For every state-action pair in the episode:
   - Calculate the return \(G\)
   - Store the return for that pair
   - Update \(Q(s,a)\) using the average return

4. Repeat the process for many episodes.

5. Display the estimated action-value function.

---

## Program

```
import numpy as np
from collections import defaultdict
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

gamma = 0.9
episodes = 50

Q = defaultdict(float)
returns = defaultdict(list)

def policy(state):
    return env.action_space.sample()

def generate_episode():
    episode = []

    state, info = env.reset()
    done = False

    while not done:
        action = policy(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        episode.append((state, action, reward))
        state = next_state

    return episode

for ep in range(episodes):
    episode = generate_episode()

    G = 0
    visited_pairs = set()

    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]

        G = gamma * G + reward

        if (state, action) not in visited_pairs:
            returns[(state, action)].append(G)
            Q[(state, action)] = np.mean(returns[(state, action)])
            visited_pairs.add((state, action))

print("\nAction Value Function:\n")

for state in range(env.observation_space.n):
    for action in range(env.action_space.n):
        print(f"State {state}, Action {action}: {Q[(state, action)]:.3f}")
```

---

## Output

```
State 0, Action 0: 0.030
State 0, Action 1: 0.027
State 0, Action 2: 0.015
State 0, Action 3: 0.000
State 1, Action 0: 0.030
State 1, Action 1: 0.000
State 1, Action 2: 0.000
State 1, Action 3: 0.000
State 2, Action 0: 0.000
State 2, Action 1: 0.000
State 2, Action 2: 0.000
State 2, Action 3: 0.000
State 3, Action 0: 0.000
State 3, Action 1: 0.000
State 3, Action 2: 0.000
State 3, Action 3: 0.000
State 4, Action 0: 0.000
State 4, Action 1: 0.070
State 4, Action 2: 0.000
State 4, Action 3: 0.000
State 5, Action 0: 0.000
State 5, Action 1: 0.000
State 5, Action 2: 0.000
State 5, Action 3: 0.000
State 6, Action 0: 0.000
State 6, Action 1: 0.000
State 6, Action 2: 0.000
State 6, Action 3: 0.000
State 7, Action 0: 0.000
State 7, Action 1: 0.000
State 7, Action 2: 0.000
State 7, Action 3: 0.000
State 8, Action 0: 0.061
State 8, Action 1: 0.000
State 8, Action 2: 0.201
State 8, Action 3: 0.000
State 9, Action 0: 0.000
State 9, Action 1: 0.266
State 9, Action 2: 0.270
State 9, Action 3: 0.000
State 10, Action 0: 0.000
State 10, Action 1: 0.450
State 10, Action 2: 0.000
State 10, Action 3: 0.000
State 11, Action 0: 0.000
State 11, Action 1: 0.000
State 11, Action 2: 0.000
State 11, Action 3: 0.000
State 12, Action 0: 0.000
State 12, Action 1: 0.000
State 12, Action 2: 0.000
State 12, Action 3: 0.000
State 13, Action 0: 0.000
State 13, Action 1: 0.729
State 13, Action 2: 0.810
State 13, Action 3: 0.000
State 14, Action 0: 0.000
State 14, Action 1: 0.900
State 14, Action 2: 1.000
State 14, Action 3: 0.000
State 15, Action 0: 0.000
State 15, Action 1: 0.000
State 15, Action 2: 0.000
State 15, Action 3: 0.000


```

## Result

Thus, the Monte Carlo Prediction algorithm was successfully implemented for estimating the action-value function \(Q(s,a)\). The expected returns for different state-action pairs were calculated using sampled episodes generated from the environment.

---


---



---

---
