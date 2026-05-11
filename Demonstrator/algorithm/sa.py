import math
import random
import helper.help_functions as helper
import numpy as np

def simulated_annealing(model, X, maxQuality, temp=10, iterations=100, neighbourhood=0.1, neighbour_index=True, schedule="exponential", alpha=0.95, converge_int=3):
    solution = help.initialize_population(X, 1)
    best_solution = solution.copy()
    solution_energy = model.predict(solution) * -1 # Speicher die Fitness 
    best_solution_energy = solution_energy
    scores = [solution_energy] # Speichere den aktuellen Wert jeder Iteration
    temp_zero = temp
    iteration_amount = 0
    for i in range(iterations):    
        if(temp <= 0 or best_solution_energy.round(converge_int) == (maxQuality * -1)):
            break
        iteration_amount = iteration_amount + 1
        if(neighbour_index):
            neighbour = get_neighbour_index(solution, neighbourhood) # Ein Nachbar von solution wird generiert
        else:
            neighbour = get_neighbour(solution, neighbourhood)

        neighbour = np.maximum(neighbour, 0) # Verhindert, dass Minuswerte entstehen
        neighbour = np.minimum(neighbour, 1)
        
        neighbour_energy = model.predict(neighbour) * -1 # Prüfe die Fitness des Nachbarn, da SA ein minimum finden soll, wird der gesuchte Max Wert zum Min Wert

        scores.append(neighbour_energy) # Ergänze die Fitness des Nachbarn zu scores
        
        # Falls entweder der Nachbar besser ist als die beste bisherige Fitness, oder
        # mit einer % Chance der SA Formel wird solution der Nachbar Wert zugewiesen
        delta = neighbour_energy - solution_energy
        alpha = random.random()
        if delta < 0 or alpha <= math.exp(-1 * (delta / temp)): 
            solution = neighbour.copy()
            solution_energy = neighbour_energy
            if (neighbour_energy < best_solution_energy) & (neighbour_energy >= (maxQuality * -1)): # Falls der Nachbar bessere Fitness hat als die momentan besterzielte, und die Fitness größer gleich maxQuality liegt
                best_solution = neighbour.copy() # Weise der Lösung den Wert von Nachbar zu
                best_solution_energy = neighbour_energy # Und die dazugehörige Fitness
        
        # Jede Iteration wird die Temperatur gesenkt
        if(schedule=="geometric"):
            temp = geometric_cooling(temp, alpha) 
        elif(schedule=="exponential"):
            temp = exponential_cooling(temp_zero, alpha, i)
        elif(schedule=="logarithmic"):
            temp = logarithmic_cooling(i, maxQuality, 1)
            print(temp)
        else:
            temp = linear_cooling(temp_zero, i)

    best_solution_energy = model.predict(best_solution) # Stelle sicher das solution_fitness nicht ein negativer Wert ist
    return best_solution, best_solution_energy, scores, iteration_amount

def linear_cooling(temperature, i):
    return temperature - i
def geometric_cooling(temperature, alpha):
    return temperature * alpha
def exponential_cooling(temperature, alpha, i):
    return temperature * (alpha**i)
def logarithmic_cooling(i, c, d=1):
    log = math.log(i + d)
    if(log != 0):
        return c / log
    else:
        return 0

# Beide Funktionen erzielen das identische, nur get_neighbour_index auf einem zufälligen Index von x und get_neighbour auf gesamten x

def get_neighbour_index(x, neighbourhood=0.1):
    neighbour = x.copy()
    index = random.randint(0, len(x) - 1) # Generiere einen zufälligen Integer in der Index Range von x
    neighbour[index] += random.uniform(-neighbourhood, neighbourhood) # Auf diesen Index addiere eine zufällige Zahl zwischen -neighbourhood und neighbourhood
    return neighbour

def get_neighbour(x, neighbourhood=0.1):
    neighbour = x.copy()
    neighbour += np.random.uniform(-neighbourhood, neighbourhood, (x.shape[1], 1)).T # Addiere zufällige Zahlen auf neighbor zwischen -neighbourhood und neighbourhood
    return neighbour