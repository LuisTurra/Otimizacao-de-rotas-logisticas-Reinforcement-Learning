import gradio as gr
import folium
import pandas as pd
import numpy as np
import os
from environment import SupplyChainEnv
from qlearning_agent import QLearningAgent

Q_TABLE_PATH = "q_table.pkl"
DQN_MODEL_PATH = "dqn_model.pt"
CSV_PATH = "data/sao_paulo_nodes.csv"


df = pd.read_csv(CSV_PATH)
print(f"CSV carregado: {len(df)} linhas")

max_ql = min(11, len(df))
df_ql = df.head(max_ql)
depot = df_ql.iloc[0].copy()
depot['demand'] = 0
locations_ql = df_ql.iloc[1:].to_dict('records')  
n_clients_ql = len(locations_ql)

print(f"[Q-Learning] Usando {n_clients_ql} clientes")

env_ql = SupplyChainEnv(locations_ql, depot, max_steps=n_clients_ql + 10)

# Carrega Q-Learning
ql_agent = None
if os.path.exists(Q_TABLE_PATH):
    ql_agent = QLearningAgent(n_actions=env_ql.action_space, env=env_ql)
    ql_agent.load(Q_TABLE_PATH)
    print("Q-Learning carregado!")
else:
    print("AVISO: q_table.pkl não encontrado.")

max_dqn = min(51, len(df))
df_dqn = df.head(max_dqn)
depot_dqn = df_dqn.iloc[0].copy()
depot_dqn['demand'] = 0
locations_dqn = df_dqn.iloc[1:].to_dict('records')  
n_clients_dqn = len(locations_dqn)

print(f"[DQN] Usando {n_clients_dqn} clientes")

env_dqn = SupplyChainEnv(locations_dqn, depot_dqn, max_steps=n_clients_dqn + 15)

# Carrega DQN
dqn_agent = None
if os.path.exists(DQN_MODEL_PATH):
    from dqn_agent import DQNAgent
    dqn_agent = DQNAgent(state_size=1 + n_clients_dqn, action_size=env_dqn.action_space)
    dqn_agent.load(DQN_MODEL_PATH)
    print("DQN carregado!")
else:
    print("AVISO: dqn_model.pt não encontrado.")

def run_dqn():
    if not dqn_agent:
        return "ERRO: dqn_model.pt não encontrado.", None, None

    local_env = SupplyChainEnv(locations_dqn, depot_dqn, max_steps=n_clients_dqn + 15)
    state = local_env.reset()
    state_vec = np.array([local_env.current_idx] + list(local_env.remaining_demand))
    total_cost = 0.0
    steps = 0
    max_steps = n_clients_dqn + 30
    route_data = [{'name': depot_dqn['name'], 'demand': 0, 'ordem': 1}]
    deliveries = 0

    while not local_env.done and steps < max_steps:
        valid = local_env._valid_actions()
        action = dqn_agent.act(state_vec, valid)
        next_state, reward, done, info = local_env.step(action)
        state_vec = np.array([local_env.current_idx] + list(local_env.remaining_demand))

        
        if len(local_env.route_positions) > 1:
            prev = local_env.route_positions[-2]
            curr = local_env.route_positions[-1]
            dist = local_env._haversine(prev, curr)
        else:
            dist = 0

        demand = local_env.original_demands[action] if action < local_env.n_locations else 0
        total_cost += dist * 1.5 + demand * 0.8

        
        if action < local_env.n_locations and local_env.remaining_demand[action] == 0:
            loc = local_env.locations[action]
            route_data.append({'name': loc['name'], 'demand': loc['demand'], 'ordem': len(route_data)})
            deliveries += 1
        elif action == local_env.n_locations and steps > 0:
            route_data.append({'name': depot_dqn['name'], 'demand': 0, 'ordem': len(route_data)})

        steps += 1

    return build_output(local_env, total_cost, route_data, "DQN", steps, deliveries, n_clients_dqn)

def run_qlearning():
    if not ql_agent:
        return "ERRO: q_table.pkl não encontrado.", None, None

    local_env = SupplyChainEnv(locations_ql, depot, max_steps=n_clients_ql + 10)
    local_agent = QLearningAgent(n_actions=local_env.action_space, env=local_env)
    local_agent.load(Q_TABLE_PATH)

    state = local_env.reset()
    total_cost = 0.0
    steps = 0
    max_steps = n_clients_ql + 20
    route_data = [{'name': depot['name'], 'demand': 0, 'ordem': 1}]
    deliveries = 0

    while not local_env.done and steps < max_steps:
        action = local_agent.choose_action(state)
        next_state, reward, done, info = local_env.step(action)

        if len(local_env.route_positions) > 1:
            prev = local_env.route_positions[-2]
            curr = local_env.route_positions[-1]
            dist = local_env._haversine(prev, curr)
        else:
            dist = 0

        demand = local_env.original_demands[action] if action < local_env.n_locations else 0
        total_cost += dist * 1.5 + demand * 0.8

        if action < local_env.n_locations and local_env.remaining_demand[action] == 0:
            loc = local_env.locations[action]
            route_data.append({'name': loc['name'], 'demand': loc['demand'], 'ordem': len(route_data)})
            deliveries += 1
        elif action == local_env.n_locations and steps > 0:
            route_data.append({'name': depot['name'], 'demand': 0, 'ordem': len(route_data)})

        state = next_state
        steps += 1

    return build_output(local_env, total_cost, route_data, "Q-Learning", steps, deliveries, n_clients_ql)

def build_output(env, total_cost, route_data, method, steps, deliveries, total_clients):
    route_df = pd.DataFrame(route_data)
    route_df = route_df[['ordem', 'name', 'demand']].rename(columns={'ordem': 'Ordem', 'name': 'Local', 'demand': 'Demanda'})

    m = folium.Map(location=[env.depot['lat'], env.depot['lon']], zoom_start=11)
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan'] * 10
    for i, pos in enumerate(env.route_positions):
        color = colors[i % len(colors)]
        popup = f"<b>{pos['name']}</b><br>Demanda: {pos.get('demand', 0)}"
        folium.CircleMarker(
            location=[pos['lat'], pos['lon']],
            radius=12 if 'depot' in pos['name'].lower() else 8,
            color=color, fill=True, popup=popup
        ).add_to(m)
        if i > 0:
            prev = env.route_positions[i-1]
            folium.PolyLine([[prev['lat'], prev['lon']], [pos['lat'], pos['lon']]], color="black", weight=3).add_to(m)

    map_html = m._repr_html_()
    return (
        f"**{method} | Custo: R$ {total_cost:,.2f}**<br>"
        f"Entregas: {deliveries}/{total_clients} | Passos: {steps}",
        route_df, map_html
    )

def show_explanation():
    return """
### **DQN vs Q-Learning**

| Característica       | **Q-Learning**                          | **DQN**                                   |
|----------------------|-----------------------------------------|-------------------------------------------|
| **Nós**              | 5–12                                    | **50–100+**                               |
| **Treinamento**      | 2–5 min                                 | 15–30 min (GPU)                           |
| **Escalabilidade**   | Baixa                                   | **Alta**                                  |

**Clique nos botões para ver!**
"""
#Gradio
with gr.Blocks(title="SupplyChain: DQN vs Q-Learning", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SupplyChain Optimizer")
    gr.Markdown("### Q-Learning (pequeno) vs DQN (grande)")

    with gr.Row():
        btn_dqn = gr.Button("DQN (Grande)", variant="primary", size="lg")
        btn_ql = gr.Button("Q-Learning (Pequeno)", variant="secondary", size="lg")
        btn_info = gr.Button("?", size="sm")

    with gr.Row():
        cost_output = gr.Markdown()

    with gr.Row():
        table = gr.Dataframe(headers=["Ordem", "Local", "Demanda"], datatype=["number", "str", "number"])

    with gr.Row():
        map_output = gr.HTML()

    with gr.Row():
        info_output = gr.Markdown()

    btn_dqn.click(fn=run_dqn, outputs=[cost_output, table, map_output])
    btn_ql.click(fn=run_qlearning, outputs=[cost_output, table, map_output])
    btn_info.click(fn=show_explanation, outputs=info_output)

    gr.Markdown("**Luís Turra** | [GitHub](https://github.com/luisturra)")

demo.launch()