"""
Resolução Unificada com ABC

INSTRUÇÕES: Comente/descomente PROBLEMA abaixo para escolher qual rodar

┌─────────────────────────────────────────────────────────────────────┐
│ TP1 - PROBLEMA 1a (Sem restrições, 2 variáveis)                    │
├─────────────────────────────────────────────────────────────────────┤
│ MIN: f(x) = 100*sqrt(|x2 - 0.01*x1^2|) + 0.01*|x1 + 10|            │
│ Limites: -15 ≤ x1 ≤ -5, -3 ≤ x2 ≤ 3                                 │
│ Variáveis: 2                                                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ TP2 - PROBLEMA 1 (Com restrições, 4 variáveis)                     │
├─────────────────────────────────────────────────────────────────────┤
│ MIN: x1^0.6 + x2^0.6 - 6x1 - 4u1 + 3u2                             │
│ Restrições: 3 (1 igualdade + 2 desigualdades)                       │
│ Variáveis: 4 [x1, x2, u1, u2]                                       │
│ NOTA: Este exemplo usa Penalidade Estática para tratar restrições  │
└─────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import matplotlib.pyplot as plt
from artificial_bee_colony import ArtificialBeeColony


# ============================================================================
#  SELEÇÃO DO PROBLEMA - COMENTE/DESCOMENTE CONFORME NECESSÁRIO 
# ============================================================================

#PROBLEMA = "TP1_1a"      # Descomente para TP1 Problema 1a
PROBLEMA = "TP2_1"     # Descomente para TP2 Problema 1


# ============================================================================
# TP1 - PROBLEMA 1a (SEM RESTRIÇÕES)
# ============================================================================

def tp1_problema_1a(x):
    """
    TP1 - Problema 1a: Função CONTÍNUA sem restrições
    
    Minimizar: f(x) = 100*sqrt(|x2 - 0.01*x1^2|) + 0.01*|x1 + 10|
    
    Limites: -15 ≤ x1 ≤ -5, -3 ≤ x2 ≤ 3
    Variáveis: 2 (x1, x2)
    """
    x1, x2 = x[0], x[1]
    
    termo1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
    termo2 = 0.01 * np.abs(x1 + 10)
    
    return termo1 + termo2


# ============================================================================
# TP2 - PROBLEMA 1 (COM RESTRIÇÕES - PENALIDADE ESTÁTICA)
# ============================================================================

def tp2_problema_1(x):
    """
    TP2 - Problema 1: Função COM RESTRIÇÕES
    
    Minimizar: x1^0.6 + x2^0.6 - 6x1 - 4u1 + 3u2
    
    Restrições:
        x2 - 3x1 - 3u1 = 0        (igualdade)
        x1 + 2u1 ≤ 4              (desigualdade 1)
        x2 + 2u2 ≤ 4              (desigualdade 2)
        x1 ≤ 3, u2 ≤ 1            (caixa)
        x1, x2, u1, u2 ≥ 0        (não-negatividade)
    
    Tratamento: PENALIDADE ESTÁTICA
    
    Variáveis: 4 (x1, x2, u1, u2)
    """
    x1, x2, u1, u2 = x[0], x[1], x[2], x[3]
    
    # ================================================================
    # FUNÇÃO OBJETIVO (sem restrições)
    # ================================================================
    fo = x1**0.6 + x2**0.6 - 6*x1 - 4*u1 + 3*u2
    
    # ================================================================
    # TRATAMENTO DE RESTRIÇÕES - PENALIDADE ESTÁTICA
    # ================================================================
    # A solução é penalizada se violar alguma restrição
    
    penalidade = 0.0
    peso_penalidade = 1e6      # Peso para penalizar violações severas
    epsilon = 0.0001           # Tolerância para restrição de igualdade
    
    # --- Restrição de Igualdade ---
    # Restrição: x2 - 3x1 - 3u1 = 0
    resto_igualdade = x2 - 3*x1 - 3*u1
    if abs(resto_igualdade) > epsilon:
        penalidade += peso_penalidade * (resto_igualdade ** 2)
    
    # --- Restrições de Desigualdade ---
    # Restrição: x1 + 2u1 ≤ 4
    resto_des1 = x1 + 2*u1 - 4
    if resto_des1 > 0:
        penalidade += peso_penalidade * (resto_des1 ** 2)
    
    # Restrição: x2 + 2u2 ≤ 4
    resto_des2 = x2 + 2*u2 - 4
    if resto_des2 > 0:
        penalidade += peso_penalidade * (resto_des2 ** 2)
    
    # --- Restrições de Caixa ---
    # Restrição: x1 ≤ 3
    if x1 > 3:
        penalidade += peso_penalidade * ((x1 - 3) ** 2)
    
    # Restrição: u2 ≤ 1
    if u2 > 1:
        penalidade += peso_penalidade * ((u2 - 1) ** 2)
    
    # --- Restrições de Não-Negatividade ---
    for i, var in enumerate([x1, x2, u1, u2]):
        if var < 0:
            penalidade += peso_penalidade * (var ** 2)
    
    # Retorna função objetivo + penalidades
    return fo + penalidade


# ============================================================================
# SELEÇÃO E CONFIGURAÇÃO DO PROBLEMA
# ============================================================================

def obter_configuracoes():
    """
    Retorna configurações baseada no problema selecionado.
    """
    
    if PROBLEMA == "TP1_1a":
        return {
            'name': "TP1 - Problema 1a",
            'function': tp1_problema_1a,
            'boundaries': [
                (-15, -5),    # x1
                (-3, 3)       # x2
            ],
            'colony_size': 40,
            'max_iterations': 300,
            'num_runs': 30,
            'minimize': True,
            'num_vars': 2,
            'descricao': "f(x) = 100*sqrt(|x2 - 0.01*x1^2|) + 0.01*|x1 + 10|",
            'var_names': ['x1', 'x2']
        }
    
    elif PROBLEMA == "TP2_1":
        return {
            'name': "TP2 - Problema 1",
            'function': tp2_problema_1,
            'boundaries': [
                (0, 3),       # x1: 0 ≤ x1 ≤ 3
                (0, 4),       # x2: 0 ≤ x2 ≤ 4 (aproximado)
                (0, 2),       # u1: 0 ≤ u1 ≤ 2 (aproximado)
                (0, 1)        # u2: 0 ≤ u2 ≤ 1
            ],
            'colony_size': 50,
            'max_iterations': 400,
            'num_runs': 30,
            'minimize': True,
            'num_vars': 4,
            'descricao': "MIN x1^0.6 + x2^0.6 - 6x1 - 4u1 + 3u2 (COM RESTRIÇÕES)",
            'var_names': ['x1', 'x2', 'u1', 'u2'],
            'restricoes': [
                "x2 - 3x1 - 3u1 = 0",
                "x1 + 2u1 ≤ 4",
                "x2 + 2u2 ≤ 4",
                "x1 ≤ 3",
                "u2 ≤ 1",
                "x1, x2, u1, u2 ≥ 0"
            ]
        }
    
    else:
        raise ValueError(f"❌ Problema desconhecido: {PROBLEMA}")


# Obtém configuração do problema selecionado
CONFIG = obter_configuracoes()


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def run_abc_single(config=CONFIG):
    """
    Executa ABC uma única vez.
    """
    abc = ArtificialBeeColony(
        objective_function=config['function'],
        boundaries=config['boundaries'],
        colony_size=config['colony_size'],
        max_iterations=config['max_iterations'],
        minimize=config['minimize'],
        seed=None
    )
    
    result = abc.fit()
    
    return {
        'best_solution': result['best_solution'],
        'best_value': result['best_value'],
        'fitness_history': result['best_fitness_history']
    }


def run_multiple_executions(config=CONFIG, num_runs=None):
    """
    Executa ABC múltiplas vezes de forma independente.
    """
    
    if num_runs is None:
        num_runs = config['num_runs']
    
    results = {
        'objective_values': [],
        'best_solutions': [],
        'fitness_histories': []
    }
    
    print(f"\n{'='*75}")
    print(f"  🐝 EXECUTANDO ABC {num_runs} VEZES PARA {config['name']}")
    print(f"{'='*75}\n")
    
    for run in range(num_runs):
        result = run_abc_single(config)
        
        results['objective_values'].append(result['best_value'])
        results['best_solutions'].append(result['best_solution'])
        results['fitness_histories'].append(result['fitness_history'])
        
        # Mostra progresso
        if (run + 1) % 10 == 0:
            print(f"  ✓ Execução {run + 1:2d}/{num_runs} completa")
            print(f"    Melhor valor: {result['best_value']:.8f}")
            var_str = ', '.join([f"{config['var_names'][i]}={val:.4f}" 
                                for i, val in enumerate(result['best_solution'])])
            print(f"    Solução: {var_str}\n")
    
    # Converte para numpy arrays
    results['objective_values'] = np.array(results['objective_values'])
    results['best_solutions'] = np.array(results['best_solutions'])
    results['fitness_histories'] = np.array(results['fitness_histories'])
    
    return results


def calculate_statistics(objective_values):
    """
    Calcula estatísticas dos resultados.
    """
    
    stats = {
        'minimum': np.min(objective_values),
        'maximum': np.max(objective_values),
        'mean': np.mean(objective_values),
        'std': np.std(objective_values),
        'median': np.median(objective_values),
        'q1': np.percentile(objective_values, 25),
        'q3': np.percentile(objective_values, 75),
    }
    
    return stats


def print_statistics(config, stats):
    """
    Imprime tabela de estatísticas em formato legível.
    """
    
    print("\n" + "="*75)
    print(f"ESTATÍSTICAS - {config['name']}")
    print("="*75)
    print(f"{'Métrica':<25} {'Valor':>35}")
    print("-"*75)
    print(f"{'Mínimo':<25} {stats['minimum']:>35.8f}")
    print(f"{'Máximo':<25} {stats['maximum']:>35.8f}")
    print(f"{'Média':<25} {stats['mean']:>35.8f}")
    print(f"{'Desvio-padrão':<25} {stats['std']:>35.8f}")
    print(f"{'Mediana':<25} {stats['median']:>35.8f}")
    print(f"{'Q1 (25%)':<25} {stats['q1']:>35.8f}")
    print(f"{'Q3 (75%)':<25} {stats['q3']:>35.8f}")
    print("="*75)


def print_best_solution(config, objective_values, best_solutions):
    """
    Imprime informações da melhor solução encontrada.
    """
    
    best_idx = np.argmin(objective_values)
    best_value = objective_values[best_idx]
    best_solution = best_solutions[best_idx]
    
    print("\n" + "="*75)
    print("MELHOR SOLUÇÃO ENCONTRADA")
    print("="*75)
    print(f"Índice da execução: {best_idx + 1}")
    print(f"Valor da função objetivo: {best_value:.8f}")
    print(f"\nVariáveis de decisão:")
    for i, (name, value) in enumerate(zip(config['var_names'], best_solution)):
        print(f"  {name} = {value:.8f}")
    
    if 'restricoes' in config:
        print(f"\nRestrições do problema:")
        for restricao in config['restricoes']:
            print(f"  • {restricao}")
    
    print("="*75)


def plot_boxplot(config, objective_values):
    """
    Gera boxplot dos resultados.
    """
    
    filename = f"boxplot_{PROBLEMA.lower()}.png"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bp = ax.boxplot(
        [objective_values],
        labels=['ABC'],
        patch_artist=True,
        widths=0.5
    )
    
    bp['boxes'][0].set_facecolor('lightblue')
    
    ax.set_ylabel('Valor da Função Objetivo', fontsize=12, fontweight='bold')
    ax.set_title(f'{config["name"]}: Distribuição de Resultados (30 execuções)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Boxplot salvo em: {filename}")
    plt.close()


def plot_convergence(config, fitness_histories):
    """
    Plota convergência média de múltiplas execuções.
    """
    
    filename = f"convergence_{PROBLEMA.lower()}.png"
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    mean_fitness = np.mean(fitness_histories, axis=0)
    std_fitness = np.std(fitness_histories, axis=0)
    
    iterations = np.arange(len(mean_fitness))
    
    ax.plot(iterations, mean_fitness, 'b-', linewidth=2.5, label='Média')
    ax.fill_between(
        iterations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.3,
        color='blue',
        label='±1 Desvio-padrão'
    )
    
    ax.set_xlabel('Iteração', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitness (melhor solução)', fontsize=12, fontweight='bold')
    ax.set_title(f'{config["name"]}: Convergência do ABC (30 execuções)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico de convergência salvo em: {filename}")
    plt.close()



# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*75)
    print(f"{CONFIG['name']}")
    print("="*75)
    print(f"\nFunção: {CONFIG['descricao']}")
    print(f"Variáveis: {CONFIG['num_vars']} {CONFIG['var_names']}")
    print(f"\nConfigurações do ABC:")
    print(f"  • Tamanho da colônia: {CONFIG['colony_size']}")
    print(f"  • Número de iterações: {CONFIG['max_iterations']}")
    print(f"  • Número de execuções: {CONFIG['num_runs']}")
    
    # Executa ABC 30 vezes
    results = run_multiple_executions(config=CONFIG)
    
    # Calcula estatísticas
    stats = calculate_statistics(results['objective_values'])
    
    # Imprime resultados
    print_statistics(CONFIG, stats)
    print_best_solution(CONFIG, results['objective_values'], results['best_solutions'])
    
    # Gera gráficos
    plot_boxplot(CONFIG, results['objective_values'])
    plot_convergence(CONFIG, results['fitness_histories'])
    
    print("\n" + "="*75)
    print("✅ Execução concluída com sucesso!")
    print("="*75 + "\n")