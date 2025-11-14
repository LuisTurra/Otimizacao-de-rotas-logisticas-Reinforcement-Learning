# SupplyChain Optimizer  
**Otimização de Rotas Logísticas em São Paulo com Q-Learning & DQN**

![DQN vs Q-Learning](https://img.shields.io/badge/DQN-50%2B%20nós-blue) ![Q-Learning](https://img.shields.io/badge/Q--Learning-10%20nós-green) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Gradio](https://img.shields.io/badge/Gradio-Interface-orange)

> **Compare dois algoritmos de Reinforcement Learning em tempo real: Q-Learning (pequeno) vs DQN (escala real)**

----------------------------------

**Demo Live** - https://huggingface.co/spaces/luisturra/otimizacao_supplaychain
**Acesse meu portfolio** - https://luisturra.github.io/MyPortfolio
-----------------------------------------------------------------------------
## Funcionalidades

- **2 Algoritmos em 1 App**:
  - **Q-Learning**: até **10 clientes**
  - **DQN (Deep Q-Network)**: até **50+ clientes**
- **Custo otimizado**: distância × R$ 1,50/km + demanda × R$ 0,80/unidade
- **Mapa interativo** com Folium (linhas + popups)
- **Interface Gradio** com 3 botões:
  - **DQN (Grande)**
  - **Q-Learning (Pequeno)**
  - ****?** - explicação didática
- **Escalável** com `min()` → funciona com qualquer tamanho de CSV
- **Deploy no Hugging Face Spaces** (gratuito)

---

## Resultados Reais (CSV com 20 nós)

| Algoritmo     | Clientes | Entregas | Custo Total | Passos |
|---------------|----------|----------|-------------|--------|
| **Q-Learning** | 10      | 10/10    | **R$ 1.250** | 11    |
| **DQN**        | 19      | 19/19    | **R$ 2.980** | 20    |

> **DQN escala 5x mais com custo proporcional — ideal para logística real.**

-------------------------------------------------------------------------------

## Tech Stack

| Tecnologia     | Uso |
|----------------|-----------------------|
| `Python`       | Lógica                |
| `Gradio`       | Interface web         |
| `Folium`       | Mapa interativo       |
| `Geopy`        | Cálculo de distância  |
| `Pandas`       | Leitura do CSV        |
| `PyTorch`      | DQN (rede neural)     |
| `Q-Learning`   | Tabela de valores     |
| `DQN`          | Rede neural profunda  |

---

## Como Rodar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/luisturra/supplychain-optimizer.git
cd supplychain-optimizer

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Coloque os arquivos na pasta:
#    - data/sao_paulo_nodes.csv
#    - q_table.pkl
#    - dqn_model.pt

# 4. Rode o app
python app.py