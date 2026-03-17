"""High-level STARLA execution entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from starla.abstraction.qvalue_abstraction import QValueAbstraction
from starla.agents.base import AgentProtocol
from starla.config import StarlaConfig
from starla.core.candidate import Candidate, Episode
from starla.core.mosa import MOSAEngine
from starla.envs.base import EnvProtocol
from starla.faults.base import FaultOracle
from starla.ml.encoder import EpisodeEncoder
from starla.ml.predictor import FaultPredictor


@dataclass(slots=True)
class TestReport:
    found_faults: dict[str, int]
    total_budget: dict[str, Any]
    archive: list[Candidate]
    generations: list[list[Candidate]]
    timing: dict[str, float]


class _SearchAbstraction:
    """Bridge abstraction object for MOSA state abstraction + episode encoding."""

    def __init__(self, abstraction: QValueAbstraction, encoder: EpisodeEncoder):
        self._abstraction = abstraction
        self._encoder = encoder

    def abstract(self, state: Any) -> tuple[Any, ...]:
        return self._abstraction.abstract(state)

    def encode(self, episode: Episode) -> Any:
        return self._encoder.encode(episode).reshape(1, -1)


class StarlaRunner:
    """Prepare data and execute STARLA end-to-end."""

    def __init__(
        self,
        config: StarlaConfig,
        agent: AgentProtocol,
        env: EnvProtocol,
        fault_oracle: FaultOracle,
    ):
        self.config = config
        self.agent = agent
        self.env = env
        self.fault_oracle = fault_oracle

        self._initial_population: list[Candidate] = []
        self._training_time: float = 0.0
        self._abstraction: QValueAbstraction | None = None
        self._encoder: EpisodeEncoder | None = None
        self._predictor: FaultPredictor | None = None

    def prepare_data(
        self,
        training_episodes: list[Episode],
        random_episodes: list[Episode],
    ) -> list[Candidate]:
        start = time.perf_counter()
        all_episodes = list(training_episodes) + list(random_episodes)
        source_for_population = random_episodes if random_episodes else training_episodes

        self._abstraction = QValueAbstraction(
            self.agent, granularity=self.config.abstraction_granularity
        )
        abstract_states_set: set[tuple[Any, ...]] = set()
        for episode in all_episodes:
            for transition in episode:
                abstract_state = self._abstraction.abstract(transition[0])
                if abstract_state == ("end",):
                    continue
                abstract_states_set.add(abstract_state)

        abstract_states = sorted(abstract_states_set, key=str)
        self._encoder = EpisodeEncoder(abstract_states, self._abstraction)
        self._predictor = FaultPredictor(self._encoder).fit(all_episodes, self.fault_oracle)

        self._initial_population = []
        for episode in source_for_population:
            candidate = Candidate(episode)
            if episode:
                first_state = episode[0][0]
                if not (isinstance(first_state, str) and first_state == "done"):
                    candidate.start_state = first_state
            self._initial_population.append(candidate)
        self._training_time = time.perf_counter() - start
        return self._initial_population

    def run(self) -> TestReport:
        if self._abstraction is None or self._encoder is None or self._predictor is None:
            raise RuntimeError("Call prepare_data(...) before run().")
        if not self._initial_population:
            raise RuntimeError("Initial population is empty. Provide non-empty episodes.")

        start = time.perf_counter()
        search_abstraction = _SearchAbstraction(self._abstraction, self._encoder)
        engine = MOSAEngine(
            config=self.config,
            agent=self.agent,
            env=self.env,
            fault_oracle=self.fault_oracle,
            abstraction=search_abstraction,
            ml_predictor=self._predictor,
        )
        result = engine.run(self._initial_population)
        run_time = time.perf_counter() - start

        functional_faults = sum(
            1 for candidate in result.archive if self.fault_oracle.is_functional_fault(candidate.episode)
        )
        reward_faults = sum(
            1 for candidate in result.archive if self.fault_oracle.is_reward_fault(candidate.episode)
        )

        return TestReport(
            found_faults={
                "functional_faults": functional_faults,
                "reward_faults": reward_faults,
                "total_archive_candidates": len(result.archive),
            },
            total_budget={
                "population_size": self.config.population_size,
                "num_generations": self.config.num_generations,
                "time_budget_seconds": self.config.time_budget_seconds,
            },
            archive=result.archive,
            generations=result.all_generations,
            timing={
                "prepare_data_seconds": self._training_time,
                "run_seconds": run_time,
                "total_seconds": self._training_time + run_time,
            },
        )

