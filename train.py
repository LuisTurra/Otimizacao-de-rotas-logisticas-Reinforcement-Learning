import pandas as pd
import numpy as np
import os
from environment import SupplyChainEnv
from qlearning_agent import QLearningAgent

Q_TABLE_PATH = "q_table.pkl"
CSV_PATH = "data/sao_paulo_nodes.csv"
EPISODES = 500  
MAX_STEPS = 30  
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


try:
    df_nodes = pd.read_csv(CSV_PATH)
    print(f"CSV carregado: {len(df_nodes)} nodes.")
    depot_data = df_nodes.iloc[0].to_dict()  
    depot = {'lat': depot_data['lat'], 'lon': depot_data['lon'], 'name': depot_data['name'], 'demand': 0}
    locations = []
    for _, row in df_nodes.iloc[1:].iterrows():
        locations.append({'lat': row['lat'], 'lon': row['lon'], 'name': row['name'], 'demand': row['demand']})
    print(f"Depot: {depot['name']}, Locations: {len(locations)}")
except FileNotFoundError:
    print("AVISO: sao_paulo_nodes.csv não encontrado. Usando dados hardcodeados (10 nodes).")
    depot = {'lat': -23.5505, 'lon': -46.6333, 'name': 'Depot Central', 'demand': 0}
    locations = [
        {'lat': -23.5614, 'lon': -46.6569, 'name': 'Moema', 'demand': 12},
        {'lat': -23.5890, 'lon': -46.6413, 'name': 'Vila Mariana', 'demand': 18},
        {'lat': -23.5500, 'lon': -46.6897, 'name': 'Pinheiros', 'demand': 25},
        {'lat': -23.5297, 'lon': -46.6273, 'name': 'Liberdade', 'demand': 8},
        {'lat': -23.5763, 'lon': -46.5889, 'name': 'Itaim Bibi', 'demand': 15},
        {'lat': -23.6078, 'lon': -46.6147, 'name': 'Santo Amaro', 'demand': 20},
        {'lat': -23.5200, 'lon': -46.6890, 'name': 'Vila Madalena', 'demand': 10},
        {'lat': -23.6000, 'lon': -46.5800, 'name': 'Brooklin', 'demand': 22},
        {'lat': -23.5800, 'lon': -46.6600, 'name': 'Jardins', 'demand': 14},
        {'lat': -23.6200, 'lon': -46.6200, 'name': 'Campo Belo', 'demand': 16}
    ]

env = SupplyChainEnv(locations, depot, max_steps=MAX_STEPS)
agent = QLearningAgent(
    n_actions=env.action_space,
    env=env,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_min=EPSILON_MIN,
    epsilon_decay=EPSILON_DECAY
)

rewards = []
print(f"Iniciando treinamento: {EPISODES} episódios, {len(locations)} locations.")

for ep in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < MAX_STEPS:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

    
    if steps >= MAX_STEPS:
        print(f"Ep {ep}: Timeout após {steps} steps.")

    agent.decay_epsilon()
    rewards.append(total_reward)

    if ep % 10 == 0:
        avg_last_10 = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
        print(f"Ep {ep}: Steps {steps}, Reward {total_reward:.2f}, Avg Last 10: {avg_last_10:.2f}, Epsilon {agent.epsilon:.3f}")
        env.render()  

avg_final = np.mean(rewards[-10:])
print(f"Treinamento concluído! Avg Reward últimos 10 eps: {avg_final:.2f}")
agent.save(Q_TABLE_PATH)
print(f"Q-table salva em {Q_TABLE_PATH}. Rode app.py pra testar!")