import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import os




def function_1(x, y):
    """Sphere Function: Minimum at (0,0) where f(x,y) = 0"""
    
    return x**2 + y**2

def function_2(x, y):
    """Rosenbrock Function: Minimum at (1,1) where f(x,y) = 0"""
    return 100 * (x**2 - y)**2 + (1 - x)**2




def crossover(p1, p2):
    """Arithmetic crossover"""
    alpha = random.random()
    c1_x = alpha * p1[0] + (1 - alpha) * p2[0]
    c1_y = alpha * p1[1] + (1 - alpha) * p2[1]
    
    alpha = random.random()
    c2_x = alpha * p1[0] + (1 - alpha) * p2[0]
    c2_y = alpha * p1[1] + (1 - alpha) * p2[1]
    return (c1_x, c1_y), (c2_x, c2_y)

def mutate(individual, bounds, prob=0.3):
    """Mutates with +/- 0.25 and bounds checking"""
    x, y = individual
    (x_min, x_max), (y_min, y_max) = bounds
    
    if random.random() < prob:
        x += random.choice([-0.25, 0.25])
        x = max(x_min, min(x, x_max))
        
    if random.random() < prob:
        y += random.choice([-0.25, 0.25])
        y = max(y_min, min(y, y_max))
        
    return (x, y)




def compute_fitnesses_for_selection(population, func):
    objs = [func(x, y) for x, y in population]
    
    return [1.0 / (1.0 + obj) for obj in objs], objs

def fps_select(population, fitnesses, count):
    total_fit = sum(fitnesses)
    probs = [f / total_fit for f in fitnesses]
    indices = np.random.choice(len(population), size=count, p=probs, replace=True)
    return [population[i] for i in indices]

def rbs_select(population, objs, count):
    sorted_indices = np.argsort(objs)[::-1] 
    ranks = np.zeros(len(population))
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank + 1
    total_rank = sum(ranks)
    probs = [r / total_rank for r in ranks]
    indices = np.random.choice(len(population), size=count, p=probs, replace=True)
    return [population[i] for i in indices]

def tournament_select(population, objs, count):
    selected = []
    for _ in range(count):
        i, j = random.sample(range(len(population)), 2)
        selected.append(population[i] if objs[i] < objs[j] else population[j])
    return selected

def truncation_survive(population, objs, count):
    sorted_indices = np.argsort(objs)
    survivors = [population[i] for i in sorted_indices[:count]]
    return survivors




def run_ea(func, bounds, parent_scheme, survival_scheme, pop_size=10, generations=40):
    (x_min, x_max), (y_min, y_max) = bounds
    population = [(random.uniform(x_min, x_max), random.uniform(y_min, y_max)) for _ in range(pop_size)]
    
    bsf_history = []
    acp_history = []
    best_so_far = float('inf')
    
    for gen in range(generations):
        parent_fitnesses, parent_objs = compute_fitnesses_for_selection(population, func)
        
        
        if parent_scheme == 'FPS':
            parents = fps_select(population, parent_fitnesses, pop_size)
        elif parent_scheme == 'RBS': 
            parents = rbs_select(population, parent_objs, pop_size)
        else: 
            parents = tournament_select(population, parent_objs, pop_size)
            
        
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i+1)%pop_size]
            c1, c2 = crossover(p1, p2)
            offspring.append(mutate(c1, bounds))
            offspring.append(mutate(c2, bounds))
            
        
        combined_pop = population + offspring
        _, combined_objs = compute_fitnesses_for_selection(combined_pop, func)
        
        if survival_scheme == 'Truncation':
            population = truncation_survive(combined_pop, combined_objs, pop_size)
        else: 
            population = tournament_select(combined_pop, combined_objs, pop_size)
            
        
        current_objs = [func(x, y) for x, y in population]
        gen_best = min(current_objs)
        gen_avg = np.mean(current_objs)
        
        if gen_best < best_so_far:
            best_so_far = gen_best
            
        bsf_history.append(best_so_far)
        acp_history.append(gen_avg)
        
    return bsf_history, acp_history




def run_experiments(func, bounds, func_name, safe_name):
    schemes = [
        ('Tournament', 'Tournament'), 
        ('Tournament', 'Truncation'), 
        ('FPS', 'Tournament'),        
        ('FPS', 'Truncation'),        
        ('RBS', 'Tournament'),        
        ('RBS', 'Truncation')         
    ]
    
    runs = 10
    generations = 40
    
    print(f"\n--- Running experiments for {func_name} ---")
    
    
    all_bsf_data = {}
    all_acp_data = {}
    csv_data = [["Generation", "Parent_Scheme", "Survival_Scheme", "Average_BSF", "Average_ACP"]]
    
    for p_scheme, s_scheme in schemes:
        print(f"Processing: {p_scheme} + {s_scheme}...")
        bsf_matrix = np.zeros((runs, generations))
        acp_matrix = np.zeros((runs, generations))
        
        for r in range(runs):
            bsf, acp = run_ea(func, bounds, p_scheme, s_scheme)
            bsf_matrix[r] = bsf
            acp_matrix[r] = acp
            
        mean_bsf = np.mean(bsf_matrix, axis=0)
        mean_acp = np.mean(acp_matrix, axis=0)
        
        label = f"{p_scheme} & {s_scheme}"
        all_bsf_data[label] = mean_bsf
        all_acp_data[label] = mean_acp
        
        
        for gen in range(generations):
            csv_data.append([gen+1, p_scheme, s_scheme, mean_bsf[gen], mean_acp[gen]])
            
        
        
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, generations+1), mean_bsf, label='Best-So-Far (BSF)', color='blue', linewidth=2)
        plt.plot(range(1, generations+1), mean_acp, label='Avg of Population (ACP)', color='red', linewidth=2, linestyle='--')
        
        plt.title(f"{func_name}\nParent: {p_scheme} | Survival: {s_scheme}")
        plt.xlabel("Generation ")
        plt.ylabel("Objective Value")
        plt.legend()
        plt.grid(True)
        
        
        filename = f"{safe_name}_{p_scheme}_{s_scheme}.png"
        plt.savefig(filename)
        plt.close() 
        
    
    csv_filename = f"{safe_name}_results.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)
        
    
    
    
    print(f"Generating combined summary graphs for {func_name}...")
    
    
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Overall Performance Comparison: {func_name}")
    
    plt.subplot(1, 2, 1) 
    for label, data in all_bsf_data.items():
        plt.plot(range(1, generations+1), data, label=label)
    plt.title("Average Best-So-Far (BSF) by Scheme")
    plt.xlabel("Generation #")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2) 
    for label, data in all_acp_data.items():
        plt.plot(range(1, generations+1), data, label=label)
    plt.title("Average of Current Population (ACP) by Scheme")
    plt.xlabel("Generation #")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{safe_name}_All_Schemes_Comparison.png")
    
    
    plt.show(block=False) 
    
    
    grand_bsf = np.mean(list(all_bsf_data.values()), axis=0)
    grand_acp = np.mean(list(all_acp_data.values()), axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, generations+1), grand_bsf, label='Overall Average BSF', color='blue', linewidth=2)
    plt.plot(range(1, generations+1), grand_acp, label='Overall Average ACP', color='red', linewidth=2, linestyle='--')
    plt.title(f"Combined BSF and ACP Grand Average: {func_name}")
    plt.xlabel("Generation #")
    plt.ylabel("Objective Value")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{safe_name}_Combined_BSF_ACP.png")
    
    print(f"Finished {func_name}! Please CLOSE the popup graph windows to proceed.")
    
    
    # plt.show(block=True) 




if __name__ == '__main__':
    
    bounds_f1 = ((-5, 5), (-5, 5))
    run_experiments(function_1, bounds_f1, "Function 1 (Sphere)", "Func1")

    
    bounds_f2 = ((-2, 2), (-1, 3))
    run_experiments(function_2, bounds_f2, "Function 2 (Rosenbrock)", "Func2")
    
    print("\nAll experiments complete! Check your folder for the .png and .csv files.")