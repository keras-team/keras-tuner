# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from copy import deepcopy

import numpy as np

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module


class GeneticEvolutionaryProcess(object):
    """Genetic Evolutionary Process with a population of chromosomes.

    Args:
        mutation_factor: Float, the factor by which the hyperparameters are
            mutated.
        crossover_factor: Float, the factor by which the hyperparameters are
            crossed over.
        seed: Optional integer, the random seed. Defaults to None.
    """

    def __init__(
        self,
        mutation_factor,
        crossover_factor,
        seed=None,
    ):
        self.mutation_factor = mutation_factor
        self.crossover_factor = crossover_factor
        self.seed = seed

    def _initialize_population(self, life: hp_module.HyperParameters):
        """Initialize the parents according to the life.

        Args:
            life: A `HyperParameters` instances for the initial life.

        Returns:
            A new mutated `HyperParameters` instances.
        """
        return self._mutate(life.copy(), mutate_force=True)

    def _mutate(
        self, chromosome: hp_module.HyperParameters, mutate_force=False
    ):
        """Mutate a chromosome by sampling from the hyperparameter space.

        Args:
            chromosome: A `HyperParameters` instance.
            mutate_force: Boolean, whether to force mutation.

        Returns:
            A mutated `HyperParameters` instance .
        """
        if random.random() < self.mutation_factor or mutate_force:
            mutated_chromosome = chromosome.copy()
            mutated_values = {
                hp.name: hp.random_sample(self.seed) for hp in chromosome.space
            }
            if self.seed is not None:
                self.seed += 1
            mutated_chromosome.values = mutated_values
            return mutated_chromosome
        return chromosome

    def _crossover(
        self,
        parent_1: hp_module.HyperParameters,
        parent_2: hp_module.HyperParameters,
    ):
        """Crossover two chromosomes.

        Args:
            parent_1: A `HyperParameters` instance.
            parent_2: A `HyperParameters` instance.

        Returns:
             A Tuple of two `HyperParameters` instances.
        """
        if random.random() < self.crossover_factor:
            # Select a random crossover point.
            crossover_point = random.randint(0, len(parent_1.space) - 1)
            parent_1_cross = parent_1.copy()
            parent_2_cross = parent_2.copy()

            # Swap the hyperparameters after the crossover point.
            for hp in parent_1.space[crossover_point:]:
                parent_1_cross.values[hp.name] = parent_2.values[hp.name]
                parent_2_cross.values[hp.name] = parent_1.values[hp.name]

            return parent_1_cross, parent_2_cross
        return parent_1, parent_2

    def _roulette_wheel_selection(self, scores, population):
        """Perform roulette wheel selection for generating a couple.

        Args:
            scores: A numpy array of scores.
            population: A list of `HyperParameters` instances.

        Returns:
            List of two `HyperParameters` instances.
        """
        # Normalize the scores, if they are not equal.
        if np.min(scores) != np.mean(scores):
            scores -= np.min(scores)
            scores /= np.sum(scores)

        # Generate two roulette wheel indices.
        parent_index_1, parent_index_2 = random.choices(
            range(len(population)), weights=scores, k=2
        )
        return population[parent_index_1], population[parent_index_2]

    def _tournament_selection(self, scores, population):
        """Perform tournament selection for generating a couple.

        Args:
            scores: A numpy array of scores.
            population: A list of `HyperParameters` instances.

        Returns:
            List of two `HyperParameters` instances.
        """
        # Generate two tournament indices.
        parent_index_1, parent_index_2 = random.choices(
            range(len(population)), k=2
        )
        if scores[parent_index_1] > scores[parent_index_2]:
            return population[parent_index_1], population[parent_index_2]
        return population[parent_index_2], population[parent_index_1]


class GeneticOptimizationOracle(oracle_module.Oracle):
    """Genetic algorithm tuner.

    This tuner uses a genetic algorithm to find the optimal hyperparameters.
    It is a black-box algorithm, which means it does not require the model
    to be compiled or trained. It works by keeping a population of models
    and training each model for a few epochs. The models that perform best
    are used to produce offspring for the next generation. A more detailed
    description of the algorithm can be found at [here](
    https://link.springer.com/article/10.1007/BF02823145
    ) and [here](
        https://github.com/clever-algorithms/CleverAlgorithms
    ).

    The `max_trials` parameter has to be calculated by the number of the
    population size and the number of generations, and the number of
    offspring times two. Because of the parent selection, the number of
    offspring, respectively, is the number of the population size  has to
    be used two times for parents_1 and parents_2.


    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        generation_size: Integer, the number of generation to evolve the
            offspring, respectively, the number of the offspring size.
            Defaults to 10.
        population_size: Integer, the number of models in the population at
            each generation. Defaults to 10.
        offspring_size: Integer, the number of offspring to produce at each
            generation. By default, the offspring size is equal to the
            population size. Defaults to None
        mutation_factor: Float, the factor by which the hyperparameters are
            mutated. Defaults to 0.9.
        crossover_factor: Float, the factor by which the hyperparameters are
            crossed over. Defaults to 0.1.
        threshold: Float, the threshold for the fitness function. If the
            fitness function is greater than the threshold, the search will
            stop. Defaults to None.
        selection_type: String, the type of selection to use for generating
            the offspring. Can be either "roulette_wheel" or "tournament".
            Defaults to "roulette_wheel".
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed
            to request hyperparameters that were not specified in
            `hyperparameters`. If not, then an error will be raised. Defaults
            to True.
    """

    def __init__(
        self,
        objective=None,
        generation_size=10,
        population_size=10,
        offspring_size=None,
        mutation_factor=0.9,
        crossover_factor=0.1,
        threshold=None,
        selection_type="tournament",
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
    ):
        self.generation_size = generation_size
        self.population_size = population_size
        self.offspring_size = offspring_size or population_size
        self.max_trials = self._make_max_trials

        super(GeneticOptimizationOracle, self).__init__(
            objective=objective,
            max_trials=self.max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            seed=seed,
        )

        self.mutation_factor = mutation_factor
        self.crossover_factor = crossover_factor
        self.threshold = threshold
        self.selection_type = selection_type

        if self.mutation_factor + self.crossover_factor > 1.0:
            raise ValueError(
                "The sum of the 'mutation_factors' and 'crossover_factors' "
                "must be less than 1.0."
            )
        if self.selection_type not in ["roulette_wheel", "tournament"]:
            raise ValueError(
                "The 'selection_type' must be either 'roulette_wheel' or "
                "'tournament'."
            )
        self._tried_so_far = set()
        self._max_collisions = 20
        self.gep = self._make_gep()
        self._make_ranges()
        self.population = {"hyperparameters": [], "scores": []}
        self.new_population = {"hyperparameters": [], "scores": []}
        self.values = {hp.name: hp.default for hp in self.get_space().space}
        self.parent_1, self.parent_2 = None, None

    @property
    def _make_max_trials(self):
        """Calculate the maximum number of trials."""
        return (
            self.population_size
            + self.generation_size * 2 * self.offspring_size
        )

    def _make_ranges(self):
        """Make the ranges for genetic optimization with respect to max trial.

        Due to the fact that 'oracle' refers to the 'max_trial',
        the ranges for the genetic optimization have to be calculated
        for:

            1.  'population_range': the number of models in the
                population.
            2.  'offspring_ranges': the number of offspring to produce
                at each generation.
            3. 'second_parent_range': the number of models to select
            4.  'generation_range': the number of generations, where the
                best models are selected and the offspring are produced.

        Note, 'second_parent_range' is part of the 'offspring_ranges',
        however, the second evaluation of the 'offspring_ranges' has
        to be calculated separately, because only one value can be
        returned once.
        """

        self.population_range = list(range(self.population_size))
        self.generation_range = list(
            range(
                self.population_size + self.offspring_size * 2,
                self.max_trials,
                self.offspring_size * 2 - 1,
            )
        )
        offspring_range = list(range(self.population_size, self.max_trials))
        self.first_parent_range = offspring_range[::2]
        self.second_parent_range = offspring_range[1::2]

    def _make_gep(self):
        """Make a genetic evolutionary process.

        Returns:
            A `GeneticEvolutionaryProcess` instance.
        """
        return GeneticEvolutionaryProcess(
            mutation_factor=self.mutation_factor,
            crossover_factor=self.crossover_factor,
            seed=self.seed,
        )

    def _check_score(self, score):
        """Check if the current score is better than the best threshold.

        Args:
            score: The current score.

        Returns:
            A `dict` with the status and the current value.
        """
        if self.threshold is not None and score <= self.threshold:
            return {
                "status": trial_module.TrialStatus.COMPLETED,
                "values": self.values,
            }

    def _get_current_score(self):
        """Get the current score.

        Returns:
            An integer value of the current score.
        """
        return self.trials[self.start_order[-1]].score

    def populate_space(self, trial_id):
        """Populate the space for the genetic algorithm.

        The population is created by randomly sampling the hyperparameters
        via mutation. The population is stored in the `hyperparameters`
        attribute. The scores are stored in the `scores` attribute.
        Next, the population is evaluated and the best models are selected
        via 'tournament' or 'roulette_wheel' selection. The best models
        will be used to crossover and mutate the offspring as new
        population. Before the next generation is created, the 'population'
        attribute is updated with the 'new_population' attributes and
        the 'new_population' attribute is reset. The process is repeated
        until the maximum number of trials is reached.

        Args:
            trial_id: The current trial ID.

        Returns:
            A dictionary of parameters for the current trial.
        """
        if len(self.start_order) > 0:
            # Start with population
            if int(self.start_order[-1]) in self.population_range:
                population = self.gep._initialize_population(
                    self.hyperparameters
                )
                self.values = population.values
                self.population["hyperparameters"].append(population)
                score = self._get_current_score()
                self.population["scores"].append(score)
                self._check_score(score)

            # Start with generation and offspring
            if int(self.start_order[-1]) in self.first_parent_range:
                if self.selection_type == "tournament":
                    (
                        self.parent_1,
                        self.parent_2,
                    ) = self.gep._tournament_selection(
                        scores=self.population["scores"],
                        population=self.population["hyperparameters"],
                    )
                else:
                    (
                        self.parent_1,
                        self.parent_2,
                    ) = self.gep._roulette_wheel_selection(
                        scores=self.population["scores"],
                        population=self.population["hyperparameters"],
                    )

                self.parent_1, self.parent_2 = self.gep._crossover(
                    parent_1=self.parent_1, parent_2=self.parent_2
                )
                self.values = self.parent_1.values
                self.new_population["hyperparameters"].append(
                    self.gep._mutate(self.parent_1)
                )
                score = self._get_current_score()
                self.new_population["scores"].append(score)
                self._check_score(score)
            # Second parent for the offspring generation to be evaluated
            elif int(self.start_order[-1]) in self.second_parent_range:
                self.values = self.parent_2.values
                self.new_population["hyperparameters"].append(
                    self.gep._mutate(self.parent_2)
                )
                score = self._get_current_score()
                self.new_population["scores"].append(score)
                self._check_score(score)

            if int(self.start_order[-1]) in self.generation_range:
                self.population = deepcopy(self.new_population)
                self.new_population = {"hyperparameters": [], "scores": []}

        if self.values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {
            "status": trial_module.TrialStatus.RUNNING,
            "values": self.values,
        }

    def get_state(self):
        """Get the state of the genetic algorithm."""
        state = super(GeneticOptimizationOracle, self).get_state()
        state.update(
            {
                "mutation_factor": self.mutation_factor,
                "crossover_factor": self.crossover_factor,
            }
        )
        return state

    def set_state(self, state):
        """Set the state of the genetic algorithm."""
        super(GeneticOptimizationOracle, self).set_state(state)
        self.mutation_factor = state["mutation_factor"]
        self.crossover_factor = state["crossover_factor"]
        self.gep = self._make_gep()


# Generate Genetic Algorithm Tuner
class GeneticOptimization(tuner_module.Tuner):
    """Genetic Optimization tuning with Genetic Evolutionary Process.

    Args:
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a `Model` instance). It is optional
            when `Tuner.run_trial()` is overriden and does not use
            `self.hypermodel`.
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        generation_size: Integer, the number of generation to evolve the
            offspring, respectively, the number of the offspring size.
            Defaults to 10.
        population_size: Integer, the number of models in the population at
            each generation. Defaults to 10.
        offspring_size: Integer, the number of offspring to produce at each
            generation. By default, the offspring size is equal to the
            population size. Defaults to None
        mutation_factor: Float, the factor by which the hyperparameters are
            mutated. Defaults to 0.9.
        crossover_factor: Float, the factor by which the hyperparameters are
            crossed over. Defaults to 0.1.
        threshold: Float, the threshold for the fitness function. If the
            fitness function is greater than the threshold, the search will
            stop. Defaults to None.
        selection_type: String, the type of selection to use for generating
            the offspring. Can be either "roulette_wheel" or "tournament".
            Defaults to "roulette_wheel".
        seed: Optional integer, the random seed.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
            see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        generation_size=10,
        population_size=10,
        offspring_size=None,
        mutation_factor=0.9,
        crossover_factor=0.1,
        threshold=None,
        selection_type="tournament",
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs
    ):
        oracle = GeneticOptimizationOracle(
            objective=objective,
            generation_size=generation_size,
            population_size=population_size,
            offspring_size=offspring_size,
            mutation_factor=mutation_factor,
            crossover_factor=crossover_factor,
            threshold=threshold,
            selection_type=selection_type,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super(
            GeneticOptimization,
            self,
        ).__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
