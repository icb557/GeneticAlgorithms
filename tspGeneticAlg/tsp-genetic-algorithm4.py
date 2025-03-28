
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import random

class GeneticTSPSolver:
    def __init__(self, cities, population_size=100, mutation_rate=0.01, generations=100):
        """
        Initialize the Genetic Algorithm for Traveling Salesman Problem
        
        :param cities: List of city coordinates
        :param population_size: Number of routes in each generation
        :param mutation_rate: Probability of mutation for each gene
        :param generations: Number of generations to evolve
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        # Distance matrix to store precalculated distances
        self.distance_matrix = self._calculate_distance_matrix()
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Track best route and fitness over generations
        self.best_route_history = []
        self.best_distance_history = []

    def _calculate_distance_matrix(self):
        """
        Precalculate distances between all cities
        
        :return: 2D numpy array of distances
        """
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                distances[i][j] = np.sqrt(
                    (self.cities[i][0] - self.cities[j][0])**2 + 
                    (self.cities[i][1] - self.cities[j][1])**2
                )
        return distances

    def _initialize_population(self):
        """
        Create initial population of random routes
        
        :return: List of routes (each route is a permutation of city indices)
        """
        population = []
        for _ in range(self.population_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(route)
        return population

    def _calculate_route_distance(self, route):
        """
        Calculate total distance of a route
        
        :param route: List of city indices
        :return: Total route distance
        """
        total_distance = 0
        for i in range(len(route)):
            current_city = route[i]
            next_city = route[(i + 1) % len(route)]
            total_distance += self.distance_matrix[current_city][next_city]
        return total_distance

    def _tournament_selection(self, population, tournament_size=5):
        """
        Select best route from a random tournament
        
        :param population: Current population of routes
        :param tournament_size: Number of routes to compete
        :return: Best route from tournament
        """
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=self._calculate_route_distance)

    def _crossover(self, parent1, parent2):
        """
        Perform order crossover
        
        :param parent1: First parent route
        :param parent2: Second parent route
        :return: Two offspring routes
        """
        # Choose crossover points
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start, len(parent1) - 1)
        
        # Create children
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent2)
        
        # Copy segment from parents
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        # Fill remaining genes
        remaining1 = [city for city in parent2 if city not in child1]
        remaining2 = [city for city in parent1 if city not in child2]
        
        # Fill non-crossover segment
        j1 = 0
        j2 = 0
        for i in range(len(parent1)):
            if child1[i] is None:
                child1[i] = remaining1[j1]
                j1 += 1
            if child2[i] is None:
                child2[i] = remaining2[j2]
                j2 += 1
        
        return child1, child2

    def _mutate(self, route):
        """
        Perform mutation by swapping two random cities
        
        :param route: Route to mutate
        :return: Mutated route
        """
        for i in range(len(route)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(route) - 1)
                route[i], route[j] = route[j], route[i]
        return route

    def solve(self):
        """
        Run the genetic algorithm to find the best route
        
        :return: Best route found
        """
        for generation in range(self.generations):
            # Create new population
            new_population = []
            
            # Elitism: Keep the best route
            best_route = min(self.population, key=self._calculate_route_distance)
            best_distance = self._calculate_route_distance(best_route)
            
            # Store best route and distance
            self.best_route_history.append(best_route)
            self.best_distance_history.append(best_distance)
            
            # Preserve best route
            new_population.append(best_route)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1 = self._tournament_selection(self.population)
                parent2 = self._tournament_selection(self.population)
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutate
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                # Add to new population
                new_population.extend([offspring1, offspring2])
            
            # Update population
            self.population = new_population[:self.population_size]
        
        # Return best route
        return min(self.population, key=self._calculate_route_distance)

    def print_route_details(self, route):
        """
        Print detailed information about the route
        
        :param route: List of city indices representing the route
        """
        print("\nDetailed Route:")
        total_distance = 0
        for i in range(len(route)):
            current_city = route[i]
            next_city = route[(i + 1) % len(route)]
            
            # Get city coordinates
            current_coords = self.cities[current_city]
            next_coords = self.cities[next_city]
            
            # Calculate distance between current and next city
            segment_distance = self.distance_matrix[current_city][next_city]
            total_distance += segment_distance
            
            print(f"Step {i+1}: City {current_city} (x: {current_coords[0]:.2f}, y: {current_coords[1]:.2f}) "
                  f"-> City {next_city} (x: {next_coords[0]:.2f}, y: {next_coords[1]:.2f}) "
                  f"Segment Distance: {segment_distance:.2f}")
        
        print(f"\nTotal Route Distance: {total_distance:.2f}")

    def visualize_results(self, route):
        """
        Create visualization of the best route, performance, and route details
        
        :param route: Best route to visualize
        """
        # Create a figure with more space for details
        plt.figure(figsize=(20, 10))
        
        # Route Visualization (first subplot)
        plt.subplot(2, 2, 1)
        
        # Plot cities
        x = [self.cities[i][0] for i in route + [route[0]]]
        y = [self.cities[i][1] for i in route + [route[0]]]
        
        plt.plot(x, y, 'ro-')
        
        # Add city number labels
        for i, city_idx in enumerate(route):
            city = self.cities[city_idx]
            plt.annotate(str(i+1), (city[0], city[1]), 
                         xytext=(5, 5), textcoords='offset points')
        
        plt.title('Best Route Found')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        
        # Performance Visualization (second subplot)
        plt.subplot(2, 2, 2)
        plt.plot(self.best_distance_history)
        plt.title('Best Distance per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        
        # Route Details Visualization (third subplot - full width)
        plt.subplot(2, 1, 2)
        plt.axis('off')  # Turn off axis
        
        # Generate route details
        route_details = self.get_route_details(route)
        
        # Display route details as text
        plt.text(0, 0.9, route_details, 
                 fontsize=10, 
                 family='monospace', 
                 verticalalignment='top')
        
        plt.title('Route Details')
        
        plt.tight_layout()
        plt.show()

    def get_route_details(self, route):
        """
        Generate detailed information about the route
        
        :param route: List of city indices representing the route
        :return: Detailed route information as a string
        """
        route_details = ["Detailed Route:\n"]
        total_distance = 0
        for i in range(len(route)):
            current_city = route[i]
            next_city = route[(i + 1) % len(route)]
            
            # Get city coordinates
            current_coords = self.cities[current_city]
            next_coords = self.cities[next_city]
            
            # Calculate distance between current and next city
            segment_distance = self.distance_matrix[current_city][next_city]
            total_distance += segment_distance
            
            route_details.append(
                f"Step {i+1}: City {current_city} "
                f"(x: {current_coords[0]:.2f}, y: {current_coords[1]:.2f}) "
                f"-> City {next_city} "
                f"(x: {next_coords[0]:.2f}, y: {next_coords[1]:.2f}) "
                f"Segment Distance: {segment_distance:.2f}"
            )
        
        route_details.append(f"\nTotal Route Distance: {total_distance:.2f}")
        return "\n".join(route_details)

def generate_cities(num_cities, seed=42):
    """
    Generate random city coordinates
    
    :param num_cities: Number of cities to generate
    :param seed: Random seed for reproducibility
    :return: List of city coordinates
    """
    np.random.seed(seed)
    return [(np.random.rand()*100, np.random.rand()*100) for _ in range(num_cities)]

def main():
    # Create application window
    root = tk.Tk()
    root.title("Traveling Salesman Genetic Algorithm")
    
    # Configure window
    root.geometry("500x500")
    
    # City count variable
    city_count = tk.IntVar(value=20)
    
    # Visualization results
    result_label = tk.StringVar()
    
    # Create a frame for route details
    details_frame = None
    
    def show_route_details(solver, best_route):
        nonlocal details_frame
        
        # Destroy existing details frame if it exists
        if details_frame:
            details_frame.destroy()
        
        # Create new details frame
        details_frame = tk.Toplevel(root)
        details_frame.title("Route Details")
        details_frame.geometry("600x400")
        
        # Create scrolled text widget
        details_text = scrolledtext.ScrolledText(
            details_frame, 
            wrap=tk.WORD, 
            width=70, 
            height=20
        )
        details_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Insert route details
        route_details = solver.get_route_details(best_route)
        details_text.insert(tk.INSERT, route_details)
        details_text.config(state=tk.DISABLED)  # Make read-only
    
    def run_algorithm():
        # Generate cities
        cities = generate_cities(city_count.get())
        
        # Create and solve TSP
        solver = GeneticTSPSolver(
            cities, 
            population_size=100, 
            mutation_rate=0.01, 
            generations=200
        )
        
        # Solve and get best route
        best_route = solver.solve()
        best_distance = solver._calculate_route_distance(best_route)
        
        # Update result label
        result_label.set(f"Best Route Distance: {best_distance:.2f}")
        
        # Visualize results
        solver.visualize_results(best_route)
        
        # Show route details in a new window
        show_route_details(solver, best_route)
    
    # Create and pack widgets
    ttk.Label(root, text="Number of Cities:").pack(pady=10)
    ttk.Spinbox(
        root, 
        from_=5, 
        to=50, 
        textvariable=city_count,
        width=10
    ).pack(pady=10)
    
    ttk.Button(
        root, 
        text="Run Genetic Algorithm", 
        command=run_algorithm
    ).pack(pady=10)
    
    ttk.Label(
        root, 
        textvariable=result_label
    ).pack(pady=10)
    
    # Start GUI
    root.mainloop()

if __name__ == "__main__":
    main()
