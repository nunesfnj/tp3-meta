"""
Artificial Bee Colony (ABC) Algorithm for Continuous Optimization

Implementação pura da classe ArtificialBeeColony para otimização contínua.

Referência:
-----------
[1] Karaboga, D. and Basturk, B., 2007. A powerful and efficient algorithm for 
    numerical function optimization: artificial bee colony (ABC) algorithm. 
    Journal of global optimization, 39(3), pp.459-471.
    DOI: https://doi.org/10.1007/s10898-007-9149-x
"""

import numpy as np
import random
from typing import Callable, List, Tuple, Dict


class ArtificialBeeColony:
    """
    Implementação do algoritmo Artificial Bee Colony (ABC) para otimização contínua.
    
    O algoritmo ABC é inspirado no comportamento forrageiro das abelhas melíferas.
    Simula três tipos de abelhas:
    - Employed Bees (Abelhas Empregadas): Exploram soluções vizinhas
    - Onlooker Bees (Abelhas Observadoras): Selecionam soluções baseado na qualidade
    - Scout Bees (Abelhas Exploradoras): Buscam novas soluções aleatórias
    """
    
    def __init__(
        self,
        objective_function: Callable,
        boundaries: List[Tuple[float, float]],
        colony_size: int = 40,
        max_iterations: int = 500,
        seed: int = None,
        minimize: bool = True,
        limit_counter_max: int = None
    ):
        """
        Inicializa o algoritmo ABC.
        
        Parâmetros:
        -----------
        objective_function : Callable
            Função a ser otimizada. Deve retornar um valor numérico.
        
        boundaries : List[Tuple[float, float]]
            Lista de tuplas com limites inferior e superior de cada variável.
            Exemplo: [(-5, 5), (-10, 10)] para problema 2D
        
        colony_size : int (padrão: 40)
            Número total de abelhas na colônia.
            Metade são abelhas empregadas, metade são observadoras.
        
        max_iterations : int (padrão: 500)
            Número máximo de iterações do algoritmo.
        
        seed : int (padrão: None)
            Semente para reprodutibilidade. Se None, usa sementes aleatórias.
        
        minimize : bool (padrão: True)
            Se True, minimiza a função. Se False, maximiza.
        
        limit_counter_max : int (padrão: None)
            Limite de iterações sem melhora antes de uma solução ser abandonada.
            Se None, usa colony_size * dimensão * 0.5
        """
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.objective_function = objective_function
        self.boundaries = boundaries
        self.colony_size = colony_size
        self.max_iterations = max_iterations
        self.minimize = minimize
        self.dimension = len(boundaries)
        self.num_employed = colony_size // 2
        self.num_onlooker = colony_size // 2
        
        # Limite de tentativas antes de uma solução ser descartada
        if limit_counter_max is None:
            self.limit_counter = int(self.colony_size * self.dimension * 0.5)
        else:
            self.limit_counter = limit_counter_max
        
        # Histórico para análise
        self.fitness_history = []
        self.best_fitness_history = []
        self.best_solution_history = []
        
    def _initialize_population(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inicializa população de soluções (fontes de alimento).
        
        Retorna:
        --------
        population : np.ndarray
            Matriz de soluções (colony_size x dimension)
        
        fitness : np.ndarray
            Vetor de fitness para cada solução
        """
        
        population = np.zeros((self.colony_size, self.dimension))
        
        for i in range(self.colony_size):
            for j in range(self.dimension):
                lower, upper = self.boundaries[j]
                population[i, j] = np.random.uniform(lower, upper)
        
        fitness = self._evaluate_fitness(population)
        return population, fitness
    
    def _evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """
        Avalia o fitness de uma população.
        
        Parâmetros:
        -----------
        population : np.ndarray
            Matriz de soluções a avaliar
        
        Retorna:
        --------
        fitness : np.ndarray
            Vetor de valores de fitness
        """
        
        fitness = np.zeros(population.shape[0])
        
        for i in range(population.shape[0]):
            try:
                value = self.objective_function(population[i])
                
                # Converte para fitness (inversão se estiver minimizando)
                if self.minimize:
                    # Para minimização: fitness = 1/(1 + f(x))
                    if value >= 0:
                        fitness[i] = 1.0 / (1.0 + value)
                    else:
                        fitness[i] = 1.0 + abs(value)
                else:
                    # Para maximização: fitness = f(x)/(1 + f(x))
                    if value >= 0:
                        fitness[i] = value / (1.0 + value)
                    else:
                        fitness[i] = 1.0 / (1.0 + abs(value))
                        
            except (ValueError, OverflowError):
                # Se a função retornar erro, atribui fitness muito baixo
                fitness[i] = -float('inf')
        
        return fitness
    
    def _employed_bee_phase(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        trial_count: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fase das abelhas empregadas.
        Cada abelha tenta melhorar sua solução atual.
        
        Parâmetros:
        -----------
        population : np.ndarray
            Soluções atuais
        
        fitness : np.ndarray
            Fitness de cada solução
        
        trial_count : np.ndarray
            Contador de tentativas sem melhora para cada solução
        
        Retorna:
        --------
        population, fitness, trial_count : tupla
        """
        
        for i in range(self.num_employed):
            # Seleciona dimensão aleatória para mutação
            j = np.random.randint(0, self.dimension)
            
            # Seleciona índice aleatório para exploração vizinha
            k = np.random.randint(0, self.colony_size)
            while k == i:
                k = np.random.randint(0, self.colony_size)
            
            # Gera nova solução (vizinhança)
            # x'_ij = x_ij + φ(x_ij - x_kj)
            phi = np.random.uniform(-1, 1)
            new_solution = population[i].copy()
            new_solution[j] = population[i, j] + phi * (population[i, j] - population[k, j])
            
            # Aplica limites
            lower, upper = self.boundaries[j]
            new_solution[j] = np.clip(new_solution[j], lower, upper)
            
            # Avalia nova solução
            new_fitness = self._evaluate_fitness(np.array([new_solution]))[0]
            
            # Seleção gulosa (greedy selection)
            if new_fitness > fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trial_count[i] = 0
            else:
                trial_count[i] += 1
        
        return population, fitness, trial_count
    
    def _onlooker_bee_phase(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        trial_count: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fase das abelhas observadoras.
        Selecionam soluções baseado na probabilidade proporcional ao fitness.
        
        Parâmetros:
        -----------
        population : np.ndarray
            Soluções atuais
        
        fitness : np.ndarray
            Fitness de cada solução
        
        trial_count : np.ndarray
            Contador de tentativas sem melhora para cada solução
        
        Retorna:
        --------
        population, fitness, trial_count : tupla
        """
        
        # Calcula probabilidades baseado no fitness
        fitness_sum = np.sum(fitness)
        probabilities = fitness / fitness_sum if fitness_sum > 0 else np.ones_like(fitness) / len(fitness)
        
        for i in range(self.num_onlooker):
            # Seleção por roleta (roulette wheel selection)
            selected_idx = np.random.choice(
                self.colony_size,
                p=probabilities
            )
            
            # Seleciona dimensão aleatória para mutação
            j = np.random.randint(0, self.dimension)
            
            # Seleciona índice aleatório para exploração vizinha
            k = np.random.randint(0, self.colony_size)
            while k == selected_idx:
                k = np.random.randint(0, self.colony_size)
            
            # Gera nova solução (vizinhança)
            phi = np.random.uniform(-1, 1)
            new_solution = population[selected_idx].copy()
            new_solution[j] = population[selected_idx, j] + phi * (population[selected_idx, j] - population[k, j])
            
            # Aplica limites
            lower, upper = self.boundaries[j]
            new_solution[j] = np.clip(new_solution[j], lower, upper)
            
            # Avalia nova solução
            new_fitness = self._evaluate_fitness(np.array([new_solution]))[0]
            
            # Seleção gulosa
            if new_fitness > fitness[selected_idx]:
                population[selected_idx] = new_solution
                fitness[selected_idx] = new_fitness
                trial_count[selected_idx] = 0
            else:
                trial_count[selected_idx] += 1
        
        return population, fitness, trial_count
    
    def _scout_bee_phase(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        trial_count: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fase das abelhas exploradoras.
        Substitui soluções abandonadas (sem melhora) por novas aleatórias.
        
        Parâmetros:
        -----------
        population : np.ndarray
            Soluções atuais
        
        fitness : np.ndarray
            Fitness de cada solução
        
        trial_count : np.ndarray
            Contador de tentativas sem melhora para cada solução
        
        Retorna:
        --------
        population, fitness, trial_count : tupla
        """
        
        for i in range(self.colony_size):
            # Se uma solução não melhorou após limit_counter tentativas
            if trial_count[i] >= self.limit_counter:
                # Substitui por nova solução aleatória
                for j in range(self.dimension):
                    lower, upper = self.boundaries[j]
                    population[i, j] = np.random.uniform(lower, upper)
                
                fitness[i] = self._evaluate_fitness(np.array([population[i]]))[0]
                trial_count[i] = 0
        
        return population, fitness, trial_count
    
    def fit(self) -> Dict:
        """
        Executa o algoritmo ABC.
        
        Retorna:
        --------
        Dict com histórico de execução
        """
        
        # Inicialização
        population, fitness = self._initialize_population()
        trial_count = np.zeros(self.colony_size)
        
        # Encontra melhor solução inicial
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx].copy()
        
        # Loop principal
        for iteration in range(self.max_iterations):
            
            # Fase 1: Abelhas Empregadas
            population, fitness, trial_count = self._employed_bee_phase(
                population, fitness, trial_count
            )
            
            # Fase 2: Abelhas Observadoras
            population, fitness, trial_count = self._onlooker_bee_phase(
                population, fitness, trial_count
            )
            
            # Fase 3: Abelhas Exploradoras
            population, fitness, trial_count = self._scout_bee_phase(
                population, fitness, trial_count
            )
            
            # Atualiza melhor solução encontrada
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()
            
            # Histórico
            self.best_fitness_history.append(best_fitness)
            self.fitness_history.append(np.mean(fitness))
            self.best_solution_history.append(best_solution.copy())
        
        # Converte fitness de volta para valor de função objetivo
        if self.minimize:
            best_value = self._fitness_to_objective(best_fitness, minimize=True)
        else:
            best_value = self._fitness_to_objective(best_fitness, minimize=False)
        
        return {
            'best_solution': best_solution,
            'best_value': best_value,
            'best_fitness': best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'population': population,
            'fitness': fitness
        }
    
    def _fitness_to_objective(self, fitness: float, minimize: bool) -> float:
        """
        Converte valor de fitness de volta para valor de função objetivo.
        """
        if minimize:
            # Inverte a transformação: fitness = 1/(1 + f(x))
            if fitness > 0:
                return (1.0 - fitness) / fitness
            else:
                return float('inf')
        else:
            # Inverte a transformação: fitness = f(x)/(1 + f(x))
            if fitness < 1.0:
                return fitness / (1.0 - fitness)
            else:
                return float('inf')
    
    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        """
        Retorna a melhor solução encontrada e seu valor de função objetivo.
        """
        if not self.best_solution_history:
            raise RuntimeError("Algoritmo ainda não foi executado. Chame fit() primeiro.")
        
        best_solution = self.best_solution_history[-1]
        best_value = self.objective_function(best_solution)
        
        return best_solution, best_value