import random
import numpy as np
import numpy.random
import helper.help_functions as helper

def ga(model, X, maxQuality, pop_size=30, generations=100, mutation_rate=1, selection="tournament", crossover="blend", converge_int=3):
    population = help.initialize_population(X, pop_size) # Initialisierung der Population
    
    # Da die Population min max Skaliert ist, sind lower und upper bound effektiv immer 0 und 1
    lower_bound = population.min(axis=0).tolist()
    upper_bound = population.max(axis=0).tolist()
    
    best_individuals = [] # Speichert das beste Individuum jeder Generation
    scores = [] # Speicher die höchste Fitness jeder Generation
    iteration_amount = 0
    for generation in range(generations):
        population = np.maximum(population, 0) # Verhindert, dass Minuswerte entstehen
        population = np.minimum(population, 1)
        
        fitness_values = np.array([model.predict(population)]) # Speichert die Fitness für jedes Individuum der Population


        best_fitness = fitness_values[fitness_values <= maxQuality] # Erhalte Fitnesswerte unter maxQuality
        if(len(best_fitness) > 0):
            best_fitness = best_fitness.max() # Leere Liste würde bei .max() crashen
            blank, best_individual = np.where(fitness_values == best_fitness) # Index von best_fitness
            if(best_individual.size > 1):
                best_individual = np.random.choice(best_individual) # Für den Fall das zwei Individuen in der Population den angestrebten höchstwert erreichen wird ein zufälliger Wert von beiden ausgewählt
        
            best_individuals.append((population[best_individual], best_fitness)) # Speichere in best_performers das beste Individuum + Fitness der Generation
            if(best_fitness.round(converge_int) == maxQuality):
                break
        
        #scores.append(fitness_values.max())
        scores.append(fitness_values.mean()) #Berechne durchschnitt
        iteration_amount = iteration_amount + 1
        # 3 Mögliche Selektionsmethoden: Roulette-, Ranked-, und Tournament Selektion. Tournament ist der Standard
        
        if(selection=="roulette"):
            population = roulette_selection(population, fitness_values)
        elif(selection=="ranked"):
            population = ranked_selection(population, fitness_values)
        else:
            population = tournament_selection(population, fitness_values)

        next_population = [] # List welche die nächste Generation speichert
        for i in range(0, len(population), 2): # Iteriere durch die gesamte Population in 2er Schritten
            
            # Wähle zwei Eltern von der iten Stelle ausgehend aus
            parent1 = population[i]
            parent2 = population[i + 1]

            # Erstelle zwei neue Individuen basierend auf der "DNA" der "Eltern"
            if(crossover=="arithmetic"):
                child1, child2 = arithmetic_crossover(parent1, parent2)
            else:
                child1, child2 = blend_crossover(parent1, parent2)
            

            # Füge die neuen Individuen zur nächsten Generation hinzu, wende Mutation an
            next_population.append(uniform_mutation(child1, mutation_rate, lower_bound, upper_bound))
            next_population.append(uniform_mutation(child2, mutation_rate, lower_bound, upper_bound))

        next_population = np.array(next_population) # List muss in np.array konvertiert werden
        next_population[0] = population[best_individual] # Behalte das beste Individuum der vorherigen Generation bei
        population = next_population # Ersetze die vorherige mit der neuen Generation
    
    best_position, best_fitness = max(best_individuals, key=lambda x: x[1]) # Speichere das beste Individuum und dessen Fitness aus best_performers, gemessen an dem Individuum mit der Größten Fitness
    best_position = best_position.reshape(1, -1) # Normalerweise hat best_position bereits diese Form, aber bei einem Datensatz mit 2 Features ist der Datensatz nicht in dieser Form wodurch die Rückskalierung bei execute_ga nicht funktioniert
    return best_position, best_fitness, scores, iteration_amount

def tournament_selection(population, fitnesses, k=3):
    pop_selected = []
    for _ in range(len(population)):
        tournament = []
        for i in range(k):
            tournament.append(random.choice(list(zip(population, fitnesses[0])))) # Es werden k viele Zufällige Listenelemente in tournament abgespeichert
        pop_selected.append(max(tournament, key=lambda x: x[1])[0]) # Individuum mit höchster Fitness wird zur selected liste hinzugefügt
    return np.array(pop_selected)

# Roulette Wheel Selection
def roulette_selection(population, fitnesses):
    fitness = np.maximum(fitnesses[0], 0) # Negative Werte werden zu 0 (Negative Werte würden als Wahrscheinlichkeit nicht funktionieren, und sollten auch im Sinne des Algorithmus nicht selektiert werden)
    maximum = np.sum(fitness)

    if maximum == 0: # Für den unwahrscheinlichen Fall das alle Fitnesswerte negativ bzw 0 sind, damit nicht durch 0 geteilt wird
        prob = np.ones(len(population)) / len(population)
    else:
        prob = fitness / maximum # Die wahrscheinlichkeit für jeden Wert in Fitness

    pop_selected = []
    for _ in range(len(population)):
        pop_selected.append(population[np.random.choice(len(population), p=prob)]) # Füge einen Wert aus der population hinzu, Wahrscheinlichkeit gewichtet nach vorher kalkulierten Probs 

    return np.array(pop_selected)   

# Ranked Selection
def ranked_selection(population, fitnesses):
    pop = list(zip(population, fitnesses[0])) #Erstelle liste mit Population und zugehöriger Fitness
    pop = sorted(pop, key=lambda x: x[1]) # Sortiere Liste nach Fitness aufsteigend
    
    prob = []
    total = 0
    for i in range(len(population)):
        prob.append(i) # Ordne jedem Wert den Rang i zu
        total = total + i # Berechne Gesamtsumme der Ränge
    
    prob = [i / total for i in prob] # Berechne eine Liste der Wahrscheinlichkeiten basierend auf Rang

    pop_selected = []
    for _ in range(len(population)):
        pop_selected.append(pop[np.random.choice(len(population), p=prob)][0]) # Mit den Rangbasierten Wahrscheinlichkeiten wird die neue Population erstellt

    return np.array(pop_selected)  

# Crossover function
def arithmetic_crossover(parent1, parent2): # Berechne "Kinder" indem durch alpha zufällig bestimmt wird wie viel das neue Individuum von seinen beiden "Eltern" erbt
    alpha = np.random.uniform(0,1)
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        child1.append(alpha*parent1[i] + (1-alpha)*parent2[i])
        child2.append(alpha*parent1[i] + (1-alpha)*parent2[i])
    return np.array(child1), np.array(child2)

def blend_crossover(parent1, parent2, alpha=0.5): #
    child1 = []
    child2 = []
    for i in range(len(parent1)):
        if(parent1[i] < parent2[i]):
            child1.append(np.random.uniform((parent1[i]-alpha*(parent2[i]-parent1[i])), parent2[i]+alpha*(parent2[i]-parent1[i])))
            child2.append(np.random.uniform((parent1[i]-alpha*(parent2[i]-parent1[i])), parent2[i]+alpha*(parent2[i]-parent1[i])))
        else:
            child1.append(np.random.uniform((parent2[i]-alpha*(parent2[i]-parent1[i])), parent1[i]+alpha*(parent2[i]-parent1[i])))
            child2.append(np.random.uniform((parent2[i]-alpha*(parent2[i]-parent1[i])), parent1[i]+alpha*(parent2[i]-parent1[i])))
    return np.array(child1), np.array(child2)

# Mutation
def uniform_mutation(individual, mutation_rate, lower_bound, upper_bound):
    for i in range(len(individual)):
        if random.random() < mutation_rate: # Chance von mutation_rate % dass eine Mutation auf den jeweiligen Parameter erfolgt
            individual[i] = random.uniform(lower_bound[i], upper_bound[i]) # Ersetze das Allel durch ein zufälliges Allel zwischen 0 und 1
    return individual