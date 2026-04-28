import numpy as np
import helper.help_functions as helper

#PSO:

def pso(model, X, maxQuality, pop_size=30, iterations=100, w=1.0, c1=1, c2=2, converge_int=3):
    #Initialisierung Pop und deren Fitness
    particles = helper.initialize_population(X, pop_size) # Initialisiere Population von größe dim * num_particles
    fitness_values = np.array([model.predict(particles)])
    #Initialisierung pbest und fitness, anfänglich identisch zu particles
    pbest = np.copy(particles)
    pbest_fitness = np.array([model.predict(particles)]) # Speicher die Fitness jedes Partikels der Population
    #Initialisierung lösung, initial ein zufälliger partikel
    solution = pbest[np.random.choice(pbest.shape[0])] # Initial ist die Lösung die die PSO zurückgibt ein zufälliger Partikel
    solution_fitness = model.predict(solution.reshape(1, -1))

    scores = [] 
    #velocities anfangs leerer array
    dim = len(X[0])
    velocities = np.zeros((pop_size, dim))
    iteration_amount = 0

    for i in range(iterations):
        #gbest Berechnung:
        gbest_fitness = fitness_values[fitness_values <= maxQuality].max() # Speichere den größten Fitnesswert der unter maxQuality liegt
        gbest = particles[np.where(fitness_values[0] == gbest_fitness)] # Update gbest zur besten Lösung der Population
        gbest = gbest[0] # Falls mehrere Werte in gbest gespeichert werden, ansonsten wird gbest mehrdimensional -> dadurch Velocity -> error 
        # Falls die neue beste Fitness besser ist als die vorher definierte im Schwarm beste Fitness, ersetze diese mit der neuen besten Fitness
        
        if gbest_fitness > solution_fitness:
            solution = gbest 
            solution_fitness = gbest_fitness
        #
        
        iteration_amount = iteration_amount + 1

        if(solution_fitness.round(converge_int) == maxQuality):
            break

        #Aktualisiere Geschwindigkeitsvektor
        for p in range(pop_size):
            # Zufällige Werte für die Berechnung des Geschwindigkeitsvektors
            r1 = np.random.uniform(0, 1, dim)
            r2 = np.random.uniform(0, 1, dim)
            velocities[p] = w * velocities[p] + c1 * r1 * (pbest[p] - particles[p]) + c2 * r2 * (gbest - particles[p]) # Berechnung des Geschwindigkeitsvektors
            particles[p] += velocities[p] # Update Partikel durch addition mit dem Geschwindigkeitsvektor
        
        particles = np.maximum(particles, 0) # Verhindert, dass Minuswerte entstehen
        particles = np.minimum(particles, 1) # Max Werte sind max werte im Wertebereich des Datensatzes
        #

        fitness_values = np.array([model.predict(particles)]) # Update das Array das alle Fitness Werte der Partikel speichert

        blank, improved_particles = np.where((fitness_values > pbest_fitness) & (fitness_values <= maxQuality)) #Speichere Index der Partikel die eine bessere Fitness aufweisen als die vorherigen Partikel. Nur falls die Fitness unter der festgelegten maxQualität liegt
        
        if(improved_particles.size == 0): # Wenn improved_indices leer ist, wird sich der Datensatz nichtmehr verbessern
            break  
        # Aktualisierung pbest
        for j in range(pop_size): # Iteriere über gesamte Population
            if(model.predict(pbest[j].reshape(1, -1)) < fitness_values[0][j]): # Update pbest dort wo sich die beste Position verbessert hat
                pbest[j] = particles[j] # Jeder Partikel i wird seine bisher beste Position gespeichert haben
        pbest_fitness = np.array([model.predict(pbest)])

        scores.append(fitness_values.mean())


    solution = solution.reshape(1, -1)
    return solution, solution_fitness, scores, iteration_amount
