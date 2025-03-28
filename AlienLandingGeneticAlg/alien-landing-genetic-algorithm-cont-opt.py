import pygame
import random
import math
import time

class LandingSafetyCriteria:
    @staticmethod
    def evaluate_landing(genome):
        """
        Comprehensive landing safety evaluation
        
        Parameters:
        genome: [Velocity, Bearing, Strength, Fuel]
        
        Returns:
        - Safety score (lower is better, 0 means perfect landing)
        - Detailed safety report
        """
        velocity, bearing, strength, fuel = genome
        
        # Safety Ranges (Optimal Landing Conditions)
        SAFE_VELOCITY = (15, 30)      # m/s
        SAFE_BEARING = (3, 7)          # degrees
        SAFE_STRENGTH = (2, 4)         # structural integrity
        SAFE_FUEL = (3, 7)             # remaining fuel percentage
        
        # Calculate deviations
        velocity_deviation = (
            0 if SAFE_VELOCITY[0] <= velocity <= SAFE_VELOCITY[1] 
            else min(abs(velocity - SAFE_VELOCITY[0]), 
                     abs(velocity - SAFE_VELOCITY[1]))
        )
        
        bearing_deviation = (
            0 if SAFE_BEARING[0] <= bearing <= SAFE_BEARING[1]
            else min(abs(bearing - SAFE_BEARING[0]), 
                     abs(bearing - SAFE_BEARING[1]))
        )
        
        strength_deviation = (
            0 if SAFE_STRENGTH[0] <= strength <= SAFE_STRENGTH[1]
            else min(abs(strength - SAFE_STRENGTH[0]), 
                     abs(strength - SAFE_STRENGTH[1]))
        )
        
        fuel_deviation = (
            0 if SAFE_FUEL[0] <= fuel <= SAFE_FUEL[1]
            else min(abs(fuel - SAFE_FUEL[0]), 
                     abs(fuel - SAFE_FUEL[1]))
        )
        
        # Total safety score (lower is better)
        safety_score = (
            velocity_deviation * 2 + 
            bearing_deviation * 3 + 
            strength_deviation * 2 + 
            fuel_deviation
        )
        
        # Detailed safety report
        safety_report = {
            "Velocity": {
                "Value": velocity,
                "Safe Range": SAFE_VELOCITY,
                "Deviation": velocity_deviation
            },
            "Bearing": {
                "Value": bearing,
                "Safe Range": SAFE_BEARING,
                "Deviation": bearing_deviation
            },
            "Strength": {
                "Value": strength,
                "Safe Range": SAFE_STRENGTH,
                "Deviation": strength_deviation
            },
            "Fuel": {
                "Value": fuel,
                "Safe Range": SAFE_FUEL,
                "Deviation": fuel_deviation
            },
            "Overall Safety Score": safety_score
        }
        
        return safety_score, safety_report

class GeneticAlgorithmSimulation:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Screen setup
        self.WIDTH = 800
        self.HEIGHT = 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Alien Landing Genetic Algorithm")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BLUE = (100, 150, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        
        # Font
        self.font = pygame.font.Font(None, 24)
        
        # Genetic Algorithm Parameters
        self.population = []
        self.generation = 0
        self.POPULATION_SIZE = 50  # Increased population size for better exploration
        self.safe_landing_threshold = 3  # Safety score threshold for safe landing
        
        # Safety Criteria
        self.safety_criteria = LandingSafetyCriteria()
        
        # Initialize population
        self.initialize_population()
        
    def initialize_population(self):
        """Create initial population of genomes"""
        self.population = []
        for _ in range(self.POPULATION_SIZE):
            # [Velocity, Bearing, Strength, Fuel]
            genome = [
                random.uniform(10, 40),   # Velocity
                random.uniform(1, 15),    # Bearing
                random.uniform(1, 15),    # Strength
                random.uniform(1, 300)    # Fuel
            ]
            self.population.append(genome)
        
    def calculate_fitness(self, genome):
        """Calculate fitness based on landing safety"""
        safety_score, _ = self.safety_criteria.evaluate_landing(genome)
        return safety_score
    
    def selection(self):
        """Select best individuals based on fitness"""
        # Sort population by fitness
        sorted_population = sorted(self.population, key=self.calculate_fitness)
        
        # Select top third
        return sorted_population[:max(2, self.POPULATION_SIZE//3)]
    
    def crossover(self, parent1, parent2):
        """Create offspring by mixing parent genes"""
        child = [
            (parent1[i] + parent2[i]) / 2 for i in range(4)
        ]
        
        # Mutation with adaptive rate
        mutation_rate = 0.3 if self.generation < 10 else 0.1
        for i in range(4):
            if random.random() < mutation_rate:
                child[i] += random.uniform(-3, 3)
        
        return child
    
    def next_generation(self):
        """Create next generation through selection and crossover"""
        # Selection
        selected = self.selection()
        
        # Create new population through crossover
        new_population = []
        while len(new_population) < self.POPULATION_SIZE:
            # Randomly pair parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Create child
            child = self.crossover(parent1, parent2)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        # Check if we've found a safe landing
        best_genome = min(self.population, key=self.calculate_fitness)
        safety_score, _ = self.safety_criteria.evaluate_landing(best_genome)
        
        return safety_score <= self.safe_landing_threshold
    
    def draw_alien_trajectory(self, genome):
        """Draw alien trajectory based on genome parameters"""
        v, b, s, f = genome
        
        # Starting point
        start_x = self.WIDTH // 2
        start_y = 50
        
        # Calculate trajectory
        angle = math.radians(90 - b * 6)  # Convert bearing to angle
        end_x = start_x + math.cos(angle) * (v * 5)
        end_y = self.HEIGHT - 100
        
        # Draw trajectory
        pygame.draw.line(self.screen, self.GREEN, 
                         (start_x, start_y), 
                         (end_x, end_y), 2)
        
        # Draw alien at end point
        pygame.draw.circle(self.screen, (255, 0, 0), 
                           (int(end_x), int(end_y)), 10)
    
    def run(self):
        """Main game loop"""
        running = True
        clock = pygame.time.Clock()
        safe_landing_found = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Manual generation advance
                        safe_landing_found = self.next_generation()
            
            # Automatically advance generations
            if not safe_landing_found:
                safe_landing_found = self.next_generation()
            
            # Clear screen
            self.screen.fill(self.BLUE)
            
            # Draw ground
            pygame.draw.line(self.screen, self.BLACK, 
                             (0, self.HEIGHT - 50), 
                             (self.WIDTH, self.HEIGHT - 50), 3)
            
            # Find best genome
            best_genome = min(self.population, key=self.calculate_fitness)
            
            # Get safety evaluation
            safety_score, safety_report = self.safety_criteria.evaluate_landing(best_genome)
            
            # Draw alien trajectory
            self.draw_alien_trajectory(best_genome)
            
            # Display generation info
            info_lines = [
                f"Generation: {self.generation}",
                f"V: {best_genome[0]:.2f} (15-30)",
                f"B: {best_genome[1]:.2f} (3-7)",
                f"S: {best_genome[2]:.2f} (2-4)",
                f"F: {best_genome[3]:.2f} (3-7)",
                f"Safety Score: {safety_score:.2f}"
            ]
            
            # Color-code safety score
            safety_color = self.GREEN
            if safety_score > 10:
                safety_color = self.RED
            elif safety_score > 3:
                safety_color = (255, 165, 0)  # Orange
            
            for i, line in enumerate(info_lines):
                text_color = safety_color if i == 5 else self.BLACK
                text = self.font.render(line, True, text_color)
                self.screen.blit(text, (self.WIDTH - 250, 50 + i*30))
            
            # If safe landing found, pause briefly
            if safe_landing_found:
                text = self.font.render("Safe Landing Achieved!", True, self.GREEN)
                self.screen.blit(text, (self.WIDTH // 2 - 100, self.HEIGHT - 25))
                pygame.display.flip()
                pygame.time.wait(3000)  # Wait for 3 seconds
                running = False
            
            # Update display
            pygame.display.flip()
            
            # Slight delay to control iteration speed
            pygame.time.delay(100)
            clock.tick(10)  # Control frame rate
        
        pygame.quit()

# Run the simulation
if __name__ == "__main__":
    simulation = GeneticAlgorithmSimulation()
    simulation.run()
