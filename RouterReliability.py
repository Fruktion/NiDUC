import random
import matplotlib.pyplot as plt
import numpy as np


class Component:
    """
    Class that represents a component of the router configuration.

    Attributes:
        name (str): Name of the component.
        failure_rate (float): The rate at which this component can fail.
        cost (float): The cost of the component.
        handling_capacity (float): The data handling capacity of the component.
        failed (bool): The status of the component, if it failed or not.
    """

    def __init__(self, name: str, failure_rate: float, cost: float, handling_capacity: float) -> None:
        self.name: str = name
        self.failure_rate: float = failure_rate
        self.cost: float = cost
        self.handling_capacity: float = handling_capacity
        self.failed: bool = False

    def fail(self, stress_factor: float = 1.0) -> bool:
        """Calculates if a component fails based on its failure_rate and a stress factor."""
        roll: float = random.uniform(0, 1)
        if roll < self.failure_rate * stress_factor:
            self.failed = True
        return self.failed


class RouterConfiguration:
    """
    Class that represents a router configuration.

    Attributes:
        name (str): Name of the configuration.
        series_components (list): List of series components.
        parallel_components (list): List of parallel components.
    """

    def __init__(self, name: str, series_components: list[Component] = [], parallel_components: list[
        Component] = []) -> None:
        self.name: str = name
        self.series_components: list[Component] = series_components
        self.parallel_components: list[Component] = parallel_components

    def add_series_component(self, component: Component) -> None:
        """Adds a series component to the router configuration."""
        self.series_components.append(component)

    def add_parallel_component(self, component: Component) -> None:
        """Adds a parallel component to the router configuration."""
        self.parallel_components.append(component)

    def fail(self, stress_factor: float) -> bool:
        """Calculates if the configuration fails based on its components."""
        return any(c.fail(stress_factor) for c in self.series_components) or any(
            c.fail(stress_factor) for c in self.parallel_components)

    def calculate_cost(self) -> float:
        """Calculates the total cost of the configuration."""
        return sum(c.cost for c in self.series_components) + sum(c.cost for c in self.parallel_components)

    def calculate_handling_capacity(self) -> float:
        """Calculates the total data handling capacity of the configuration."""
        return sum(c.handling_capacity for c in self.series_components) + sum(
            c.handling_capacity for c in self.parallel_components)


class Router:
    """
    Class that represents a router.

    Attributes:
        model (str): Model name of the router.
        configurations (list): List of possible configurations for the router.
        network_traffic (list): Network traffic data.
    """

    def __init__(self, model: str) -> None:
        self.model: str = model
        self.configurations: list[RouterConfiguration] = []
        self.network_traffic: list = []

    def add_configuration(self, configuration: RouterConfiguration) -> None:
        """Adds a configuration to the router."""
        self.configurations.append(configuration)

    def set_network_traffic(self, network_traffic: list) -> None:
        """Sets the network traffic data."""
        self.network_traffic = network_traffic

    def simulate(self, num_runs: int) -> None:
        """Simulates the router operation with all its configurations."""
        timestamps = [i for i in range(num_runs)]
        for configuration in self.configurations:
            failures = 0
            failure_rates = []
            for i in range(num_runs):
                stress_factor = 1.0 + (self.network_traffic[i] / 100.0)
                if configuration.fail(stress_factor):
                    failures += 1
                failure_rates.append(failures / (i + 1))
            self.generate_graph(configuration.name, timestamps, failure_rates)

    def generate_graph(self, name: str, timestamps: list, failure_rates: list) -> None:
        """Generates a graph for the failure rates of the router configurations."""
        plt.plot(timestamps, failure_rates)
        plt.xlabel("Time")
        plt.ylabel("Router Failure Rate")
        plt.title(f"Reliability Simulation for {self.model} - {name}")
        plt.show()


def calculate_fitness(config: RouterConfiguration, max_cost: float) -> tuple[float, float]:
    """Calculates the fitness of a configuration based on its cost and data handling capacity."""
    return config.calculate_cost() / max_cost, config.calculate_handling_capacity()


def generate_population(series_components: list[Component], parallel_components: list[
    Component], population_size: int) -> list[RouterConfiguration]:
    """Generates a population of router configurations."""
    return [
        RouterConfiguration(f"Configuration {i + 1}", random.sample(series_components, random.randint(1, len(series_components))), random.sample(parallel_components, random.randint(1, len(parallel_components))))
        for i in range(population_size)]


def select_parents(population: list[RouterConfiguration], max_cost: float) -> tuple[
    RouterConfiguration, RouterConfiguration]:
    """Selects two parent configurations based on their fitness."""
    fitnesses = [calculate_fitness(c, max_cost) for c in population]
    parents = random.choices(population, weights=[1.0 / f[0] for f in fitnesses], k=2)
    return parents[0], parents[1]


def crossover(parent1: RouterConfiguration, parent2: RouterConfiguration) -> tuple[
    RouterConfiguration, RouterConfiguration]:
    """Performs a crossover operation between two configurations."""
    series_components1 = parent1.series_components
    series_components2 = parent2.series_components
    parallel_components1 = parent1.parallel_components
    parallel_components2 = parent2.parallel_components
    if min(len(series_components1), len(series_components2)) > 1:
        split = random.randint(1, min(len(series_components1), len(series_components2)) - 1)
        offspring1 = RouterConfiguration("Offspring 1", series_components1[:split] + series_components2[
                                                                                     split:], parallel_components1[
                                                                                              :split] + parallel_components2[
                                                                                                        split:])
        offspring2 = RouterConfiguration("Offspring 2", series_components2[:split] + series_components1[
                                                                                     split:], parallel_components2[
                                                                                              :split] + parallel_components1[
                                                                                                        split:])
        return offspring1, offspring2
    else:
        return parent1, parent2


def mutate(offspring: RouterConfiguration, series_components: list[Component], parallel_components: list[
    Component], mutation_rate: float) -> RouterConfiguration:
    """Performs a mutation operation in a configuration."""
    if random.random() < mutation_rate:
        if random.random() < 0.5 and len(offspring.series_components) > 1:
            offspring.series_components.pop(random.randint(0, len(offspring.series_components) - 1))
        else:
            offspring.add_series_component(random.choice(series_components))

        if random.random() < 0.5 and len(offspring.parallel_components) > 1:
            offspring.parallel_components.pop(random.randint(0, len(offspring.parallel_components) - 1))
        else:
            offspring.add_parallel_component(random.choice(parallel_components))
    return offspring


def genetic_algorithm(router: Router, series_components: list[Component], parallel_components: list[
    Component], population_size: int, num_generations: int, mutation_rate: float, max_cost: float) -> RouterConfiguration:
    """Performs a genetic algorithm to find the optimal router configuration."""
    population = generate_population(series_components, parallel_components, population_size)
    for _ in range(num_generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, max_cost)
            offspring1, offspring2 = crossover(parent1, parent2)
            offspring1 = mutate(offspring1, series_components, parallel_components, mutation_rate)
            offspring2 = mutate(offspring2, series_components, parallel_components, mutation_rate)
            new_population.extend([offspring1, offspring2])
        population = new_population

    best_configuration = max(population, key=lambda c: calculate_fitness(c, max_cost)[1])
    router.add_configuration(best_configuration)
    return best_configuration


if __name__ == "__main__":
    # Initialize the components with their corresponding parameters
    power_supply = Component("Power Supply", 0.005, 100, 50)
    cpu = Component("CPU", 0.02, 200, 100)
    memory = Component("Memory", 0.01, 150, 80)
    interface = Component("Interface", 0.01, 50, 40)
    antenna = Component("Antenna", 0.005, 50, 20)
    firmware = Component("Firmware", 0.02, 200, 100)
    controller = Component("Controller", 0.01, 150, 80)

    # List of components that could be placed in series or parallel in a router configuration
    series_components = [power_supply, cpu, memory, interface, antenna]
    parallel_components = [firmware, controller]

    # Initialize a router model
    router = Router("Model X")

    # Set network traffic for the router model
    router.set_network_traffic([random.randint(50, 150) for _ in range(100)])

    # Run the genetic algorithm to get the best router configuration based on the specified parameters
    best_config = genetic_algorithm(router, series_components, parallel_components, 100, 100, 0.05, 500)

    # Print out the details of the best configuration
    print(f"The best configuration is {best_config.name} with a cost of {best_config.calculate_cost()} and a handling capacity of {best_config.calculate_handling_capacity()}.")
    print("It contains the following series components: " + ", ".join([c.name for c in best_config.series_components]))
    print("And the following parallel components: " + ", ".join([c.name for c in best_config.parallel_components]))

    # Run the simulation on the router with the best configuration
    router.simulate(100)
