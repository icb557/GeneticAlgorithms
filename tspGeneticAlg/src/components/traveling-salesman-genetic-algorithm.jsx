import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

// City generation utility
const generateCities = (numCities) => {
  return Array.from({ length: numCities }, (_, i) => ({
    id: i,
    x: Math.random() * 500,
    y: Math.random() * 500,
  }));
};

// Calculate distance between two cities
const calculateDistance = (city1, city2) => {
  return Math.sqrt(
    Math.pow(city1.x - city2.x, 2) + Math.pow(city1.y - city2.y, 2)
  );
};

// Genetic Algorithm Core
class GeneticTSP {
  cities;
  populationSize;
  mutationRate;
  population;

  constructor(cities, populationSize = 100, mutationRate = 0.01) {
    this.cities = cities;
    this.populationSize = populationSize;
    this.mutationRate = mutationRate;
    this.population = this.initializePopulation();
  }

  // Initialize random population of routes
  initializePopulation() {
    return Array.from({ length: this.populationSize }, () =>
      this.shuffleArray([...this.cities.map((city) => city.id)])
    );
  }

  // Shuffle array utility
  shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  // Calculate total route distance
  calculateRouteDistance(route) {
    let totalDistance = 0;
    for (let i = 0; i < route.length - 1; i++) {
      const city1 = this.cities[route[i]];
      const city2 = this.cities[route[i + 1]];
      totalDistance += calculateDistance(city1, city2);
    }
    // Add distance back to start to complete the tour
    const firstCity = this.cities[route[0]];
    const lastCity = this.cities[route[route.length - 1]];
    totalDistance += calculateDistance(lastCity, firstCity);
    return totalDistance;
  }

  // Selection using tournament selection
  selection(population) {
    const tournamentSize = 5;
    const selected = [];

    for (let i = 0; i < this.populationSize; i++) {
      const tournament = Array.from(
        { length: tournamentSize },
        () => population[Math.floor(Math.random() * population.length)]
      );

      const best = tournament.reduce((best, current) =>
        this.calculateRouteDistance(best) < this.calculateRouteDistance(current)
          ? best
          : current
      );

      selected.push(best);
    }

    return selected;
  }

  // Crossover (Order Crossover)
  crossover(parent1, parent2) {
    const start = Math.floor(Math.random() * parent1.length);
    const end = Math.floor(Math.random() * (parent1.length - start)) + start;

    const child1 = new Array(parent1.length).fill(null);
    const child2 = new Array(parent2.length).fill(null);

    // Copy segment from first parent
    for (let i = start; i <= end; i++) {
      child1[i] = parent1[i];
      child2[i] = parent2[i];
    }

    // Fill remaining with other parent's genes
    let j1 = (end + 1) % parent1.length;
    let j2 = (end + 1) % parent2.length;

    for (let i = 0; i < parent1.length; i++) {
      const idx = (end + 1 + i) % parent1.length;

      if (!child1.includes(parent2[idx])) {
        child1[j1] = parent2[idx];
        j1 = (j1 + 1) % parent1.length;
      }

      if (!child2.includes(parent1[idx])) {
        child2[j2] = parent1[idx];
        j2 = (j2 + 1) % parent2.length;
      }
    }

    return [child1, child2];
  }

  // Mutation (swap mutation)
  mutate(route) {
    const mutatedRoute = [...route];

    for (let i = 0; i < mutatedRoute.length; i++) {
      if (Math.random() < this.mutationRate) {
        const j = Math.floor(Math.random() * mutatedRoute.length);
        [mutatedRoute[i], mutatedRoute[j]] = [mutatedRoute[j], mutatedRoute[i]];
      }
    }

    return mutatedRoute;
  }

  // Run genetic algorithm
  evolve() {
    // Selection
    const selected = this.selection(this.population);

    // Crossover
    const offspring = [];
    for (let i = 0; i < selected.length; i += 2) {
      if (i + 1 < selected.length) {
        const [child1, child2] = this.crossover(selected[i], selected[i + 1]);
        offspring.push(this.mutate(child1), this.mutate(child2));
      }
    }

    // Replace population
    this.population = offspring;

    // Find best route
    return this.population.reduce((best, current) =>
      this.calculateRouteDistance(best) < this.calculateRouteDistance(current)
        ? best
        : current
    );
  }
}


// Main Component
const TravelingSalesmanGA = () => {
  const [cities, setCities] = useState([]);
  const [numCities, setNumCities] = useState(10);
  const [bestRoute, setBestRoute] = useState([]);
  const [bestDistance, setBestDistance] = useState(Infinity);
  const [generationHistory, setGenerationHistory] = useState([]);

  const runGeneticAlgorithm = () => {
    // Generate cities
    const generatedCities = generateCities(numCities);
    setCities(generatedCities);

    // Create GA instance
    const ga = new GeneticTSP(generatedCities, 100, 0.01);

    // Run multiple generations
    const history = [];
    let bestSoFar = Infinity;

    for (let gen = 0; gen < 100; gen++) {
      const currentBestRoute = ga.evolve();
      const currentBestDistance = ga.calculateRouteDistance(currentBestRoute);

      history.push({
        generation: gen,
        bestDistance: currentBestDistance,
      });

      if (currentBestDistance < bestSoFar) {
        bestSoFar = currentBestDistance;
        setBestRoute(currentBestRoute);
        setBestDistance(currentBestDistance);
      }
    }

    setGenerationHistory(history);
  };

  return (
    // Add bg-white to set white background for the entire app container
    <div className="p-4 max-w-6xl mx-auto bg-white min-h-screen">
      <Card>
        <CardHeader>
          <CardTitle>Traveling Salesman Genetic Algorithm</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex space-x-4 mb-4">
            <Input
              type="number"
              value={numCities}
              onChange={(e) => setNumCities(Number(e.target.value))}
              placeholder="Number of Cities"
              className="w-40"
            />
            <Button onClick={runGeneticAlgorithm}>Run Genetic Algorithm</Button>
          </div>

          {/* Fix the layout by removing bg-white (already white from parent) 
              and adjust the flex properties for better spacing */}
          <div className="w-full p-4 flex flex-row justify-between gap-4">
            {/* City Visualization (Left) */}
            <div className="w-1/2 border rounded-lg p-2">
              <svg
                viewBox="0 0 500 500"
                className="w-full h-[300px]"
              >
                {cities.map((city, index) => (
                  <g key={city.id}>
                    <circle cx={city.x} cy={city.y} r="5" fill="blue" />
                    <text x={city.x + 5} y={city.y - 5} fontSize="10">
                      {index}
                    </text>
                  </g>
                ))}
                {bestRoute.length > 0 && (
                  <polyline
                    points={
                      bestRoute
                        .map((cityId) => {
                          const city = cities[cityId];
                          return `${city.x},${city.y}`;
                        })
                        .join(" ") +
                      ` ${cities[bestRoute[0]].x},${cities[bestRoute[0]].y}`
                    }
                    fill="none"
                    stroke="red"
                    strokeWidth="2"
                  />
                )}
              </svg>
            </div>

            {/* Performance Chart (Right) */}
            <div className="w-1/2 border rounded-lg p-2">
              {generationHistory.length > 0 && (
                <LineChart width={800} height={600} data={generationHistory} className="mx-auto">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="generation" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="bestDistance"
                    stroke="#8884d8"
                    name="Best Route Distance"
                  />
                </LineChart>
              )}
            </div>
          </div>

          {/* Result Display */}
          {bestRoute.length > 0 && (
            <div className="mt-4">
              <p>Best Route: {bestRoute.join(" → ")}</p>
              <p>Total Distance: {bestDistance.toFixed(2)}</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TravelingSalesmanGA;
