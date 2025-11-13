# app.py
import gradio as gr
import folium
import pandas as pd
import numpy as np
import os
from environment import SupplyChainEnv
from qlearning_agent import QLearningAgent


Q_TABLE_PATH = "q_table.pkl"
CSV_PATH = "data/sao_paulo_nodes.csv"


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
    print("AVISO: sao_paulo_nodes.csv não encontrado. Usando dados hardcodeados.")
   
    depot = {'lat': -23.5505, 'lon': -46.6333, 'name': 'Depot Central', 'demand': 0}
    locations = [
        {'lat': -23.5614, 'lon': -46.6569, 'name': 'Moema', 'demand': 12},
        {'lat': -23.5890, 'lon': -46.6413, 'name': 'Vila Mariana', 'demand': 18},
        {'lat': -23.5500, 'lon': -46.6897, 'name': 'Pinheiros', 'demand': 25},
        
    ]


env = SupplyChainEnv(locations, depot, max_steps=30)  
agent = QLearningAgent(n_actions=env.action_space, env=env)


if os.path.exists(Q_TABLE_PATH) and os.path.getsize(Q_TABLE_PATH) > 0:
    try:
        agent.load(Q_TABLE_PATH)
        print("Q-table carregada com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar Q-table: {e}")
else:
    print("AVISO: q_table.pkl não encontrado. Execute train.py primeiro.")


def optimize_route():
    if not os.path.exists(Q_TABLE_PATH):
        return "ERRO: q_table.pkl não encontrado. Execute train.py.", None, None

    local_env = SupplyChainEnv(locations, depot, max_steps=30)
    local_agent = QLearningAgent(n_actions=local_env.action_space, env=local_env)
    local_agent.load(Q_TABLE_PATH)

    state = local_env.reset()
    total_cost = 0.0
    steps = 0
    max_steps = 30  
    done = False
    route_data = []  

   
    route_data.append({'name': depot['name'], 'demand': 0, 'ordem': 1})

    while not done and steps < max_steps:
        action = local_agent.choose_action(state)
        next_state, reward, done, info = local_env.step(action)
        steps += 1

        
        dist = info.get('dist', 0)
        demand = info.get('demand_original', 0)
        if info.get('delivery_made', False) or action == local_env.n_locations:
            cost_step = dist * 1.5 + demand * 0.8
            total_cost += cost_step

        
        if info.get('delivery_made', False) and 'action' in info:
            loc = local_env.locations[info['action']]
            route_data.append({'name': loc['name'], 'demand': loc['demand'], 'ordem': len(route_data) + 1})
        elif action == local_env.n_locations and steps > 1:
            route_data.append({'name': depot['name'], 'demand': 0, 'ordem': len(route_data) + 1})

        state = next_state

    if steps >= max_steps:
        return f"ERRO: Loop travado após {steps} passos.", None, None


    route_df = pd.DataFrame(route_data)
    route_df = route_df[['ordem', 'name', 'demand']].rename(columns={'ordem': 'Ordem', 'name': 'Local', 'demand': 'Demanda'})

    
    m = folium.Map(location=[depot['lat'], depot['lon']], zoom_start=10)  
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'lime', 'navy', 'teal', 'maroon', 'fuchsia', 'aqua', 'silver', 'black']
    route_positions = local_env.route_positions

    for i, pos in enumerate(route_positions):
        color = colors[i % len(colors)]
        popup_text = f"<b>{pos['name']}</b><br>Demanda: {pos.get('demand', 0)}"
        folium.CircleMarker(
            location=[pos['lat'], pos['lon']],
            radius=12 if pos['name'] == depot['name'] else 8,  
            color=color,
            fill=True,
            popup=popup_text
        ).add_to(m)
        if i > 0:
            prev_pos = route_positions[i-1]
            folium.PolyLine(
                [[prev_pos['lat'], prev_pos['lon']], [pos['lat'], pos['lon']]],
                color="black", weight=4, opacity=0.8
            ).add_to(m)

    map_html = m._repr_html_()

    return f"**Custo Total Otimizado: R$ {total_cost:,.2f}** (Rota: {len(route_data)} paradas em {steps} steps | {len(locations)} nodes totais)", route_df, map_html

with gr.Blocks(title="SupplyChain Optimizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SupplyChain Optimizer")
    gr.Markdown("### Rotas logísticas otimizadas em São Paulo com **Q-Learning** (Dados: sao_paulo_nodes.csv)")

    with gr.Row():
        btn = gr.Button("Gerar Rota Otimizada", variant="primary", size="lg")

    with gr.Row():
        cost_output = gr.Markdown()

    with gr.Row():
        table = gr.Dataframe(
            headers=["Ordem", "Local", "Demanda"],
            datatype=["number", "str", "number"]
        )

    with gr.Row():
        map_output = gr.HTML()

    btn.click(
        fn=optimize_route,
        inputs=[],
        outputs=[cost_output, table, map_output]
    )

    gr.Markdown("---")
    gr.Markdown("**Luís Turra** | [GitHub](https://github.com/luisturra) | [Portfólio](https://luisturra.github.io/MyPortfolio/)")

if __name__ == "__main__":
    demo.launch()