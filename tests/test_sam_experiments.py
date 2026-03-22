"""Lightweight unit tests for Atari / MuJoCo SAM experiment helpers (no RL, no GPU)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# --- repo + STARLA on path (for package imports; fault oracles use a stub base if needed) ---
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "STARLA" / "src"))


def _install_minimal_starla_fault_base_if_missing() -> None:
    """Allow importing fault oracle modules without full STARLA installed."""
    key = "starla.faults.base"
    try:
        __import__(key)
        return
    except ImportError:
        pass
    from abc import ABC, abstractmethod

    class FaultOracle(ABC):
        @abstractmethod
        def is_functional_fault(self, episode: list[tuple[Any, ...]]) -> bool:
            ...

        @abstractmethod
        def is_reward_fault(self, episode: list[tuple[Any, ...]]) -> bool:
            ...

        @abstractmethod
        def get_fault_thresholds(self) -> dict[str, Any]:
            ...

    base_mod = types.ModuleType(key)
    base_mod.FaultOracle = FaultOracle
    sys.modules["starla"] = types.ModuleType("starla")
    sys.modules["starla.faults"] = types.ModuleType("starla.faults")
    sys.modules[key] = base_mod


_install_minimal_starla_fault_base_if_missing()

try:
    from experiments_atari_sam.fault_oracle.breakout_fault_oracle import BreakoutFaultOracle
    from experiments_mujoco_sam.fault_oracle.halfcheetah_fault_oracle import HalfCheetahFaultOracle
except ImportError as exc:  # pragma: no cover - fault oracle imports must not fail in CI
    raise AssertionError("Fault oracle modules must import with stub starla only") from exc

# SAM adapters pull in experiments_sam.sam_mutation → torch; mock before import.
sys.modules.setdefault("torch", MagicMock())
from experiments_atari_sam.adapters.sam_mutation_atari import SAMGuidedMutatorAtari
from experiments_mujoco_sam.adapters.sam_mutation_mujoco import SAMGuidedMutatorMujoco


# =============================================================================
# Helpers — episode layout: [(obs, action), ..., (last_obs, -1), ("done", total_reward)]
# =============================================================================


def make_breakout_episode(n_steps: int, total_reward: float) -> list[tuple[Any, ...]]:
    body: list[tuple[Any, ...]] = [
        (np.zeros((3, 64, 64), dtype=np.float32), 0) for _ in range(n_steps)
    ]
    return body + [(np.zeros((3, 64, 64), dtype=np.float32), -1), ("done", total_reward)]


def make_cheetah_episode(n_steps: int, total_reward: float) -> list[tuple[Any, ...]]:
    body: list[tuple[Any, ...]] = [
        (np.zeros(17, dtype=np.float32), np.zeros(6, dtype=np.float32)) for _ in range(n_steps)
    ]
    return body + [(np.zeros(17, dtype=np.float32), -1), ("done", total_reward)]


# =============================================================================
# Part 1 — BreakoutFaultOracle
# =============================================================================


def test_breakout_functional_fault_short_episode() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(30, 100.0)
    assert o.is_functional_fault(ep) is True


def test_breakout_functional_fault_normal_episode() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(100, 100.0)
    assert o.is_functional_fault(ep) is False


def test_breakout_functional_fault_nonfinite_obs() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(100, 100.0)
    ep[50] = (np.full((3, 64, 64), np.nan, dtype=np.float32), 0)
    assert o.is_functional_fault(ep) is True


def test_breakout_reward_fault_below_threshold() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(100, 2.0)
    assert o.is_reward_fault(ep) is True


def test_breakout_reward_fault_above_threshold() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(100, 10.0)
    assert o.is_reward_fault(ep) is False


def test_breakout_reward_fault_missing_done() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(100, 2.0)
    ep[-1] = ("finished", 2.0)
    assert o.is_reward_fault(ep) is False


def test_breakout_strict_fault_and_logic() -> None:
    o = BreakoutFaultOracle()
    ep_and = make_breakout_episode(30, 3.0)
    assert o.is_functional_fault_strict(ep_and) is True
    ep_and2 = make_breakout_episode(30, 20.0)
    assert o.is_functional_fault_strict(ep_and2) is False


def test_breakout_legacy_fault() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(100, 100.0)
    last = np.zeros((3, 64, 64), dtype=np.float32)
    last.flat[0] = np.inf
    ep[-2] = (last, -1)
    assert o.is_functional_fault_legacy(ep) is True


def test_breakout_window_fault() -> None:
    o = BreakoutFaultOracle()
    ep = make_breakout_episode(100, 100.0)
    bad = np.zeros((3, 64, 64), dtype=np.float32)
    bad.flat[0] = np.nan
    ep[98] = (bad, 0)
    assert o.is_functional_fault_window(ep) is True


def test_breakout_get_thresholds() -> None:
    o = BreakoutFaultOracle()
    d = o.get_fault_thresholds()
    assert isinstance(d, dict)
    assert "reward_fault_threshold" in d
    assert "min_steps" in d


# =============================================================================
# Part 2 — HalfCheetahFaultOracle
# =============================================================================


def test_cheetah_functional_fault_short_episode() -> None:
    o = HalfCheetahFaultOracle()
    ep = make_cheetah_episode(50, 1000.0)
    assert o.is_functional_fault(ep) is True


def test_cheetah_functional_fault_normal_episode() -> None:
    o = HalfCheetahFaultOracle()
    ep = make_cheetah_episode(200, 5000.0)
    assert o.is_functional_fault(ep) is False


def test_cheetah_reward_fault_below_threshold() -> None:
    o = HalfCheetahFaultOracle()
    ep = make_cheetah_episode(200, 50.0)
    assert o.is_reward_fault(ep) is True


def test_cheetah_reward_fault_above_threshold() -> None:
    o = HalfCheetahFaultOracle()
    ep = make_cheetah_episode(200, 500.0)
    assert o.is_reward_fault(ep) is False


def test_cheetah_strict_fault_or_logic() -> None:
    o = HalfCheetahFaultOracle()
    assert o.is_functional_fault_strict(make_cheetah_episode(50, 1000.0)) is True
    assert o.is_functional_fault_strict(make_cheetah_episode(200, 50.0)) is True
    assert o.is_functional_fault_strict(make_cheetah_episode(200, 1000.0)) is False


# =============================================================================
# Part 3 — SAMGuidedMutatorAtari._random_fallback
# =============================================================================


def test_atari_fallback_float32_input() -> None:
    np.random.seed(0)
    inp = np.random.rand(3, 64, 64).astype(np.float32)
    out = SAMGuidedMutatorAtari._random_fallback(inp)
    assert out.shape == inp.shape
    assert out.dtype == np.float32
    assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0


def test_atari_fallback_uint8_input() -> None:
    np.random.seed(1)
    inp = np.random.randint(0, 256, (3, 64, 64), dtype=np.uint8)
    out = SAMGuidedMutatorAtari._random_fallback(inp)
    assert out.shape == inp.shape
    assert out.dtype == np.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_atari_fallback_not_identical_to_input() -> None:
    np.random.seed(2)
    inp = np.random.rand(3, 64, 64).astype(np.float32)
    out = SAMGuidedMutatorAtari._random_fallback(inp)
    assert not np.array_equal(out, inp)


# =============================================================================
# Part 4 — SAMGuidedMutatorMujoco._random_fallback
# =============================================================================


def test_mujoco_fallback_numpy_input() -> None:
    np.random.seed(3)
    inp = np.ones(17, dtype=np.float32)
    out = SAMGuidedMutatorMujoco._random_fallback(inp)
    assert out.shape == (17,)
    assert out.dtype == np.float32
    assert not np.allclose(out, 1.0)


def test_mujoco_fallback_preserves_shape() -> None:
    for shp in ((17,), (4,), (1,)):
        inp = np.ones(shp, dtype=np.float32)
        out = SAMGuidedMutatorMujoco._random_fallback(inp)
        assert out.shape == shp


def test_mujoco_fallback_list_input() -> None:
    inp = [1.0] * 17
    out = SAMGuidedMutatorMujoco._random_fallback(inp)
    assert isinstance(out, list)


def test_mujoco_fallback_tuple_input() -> None:
    inp = tuple([1.0] * 4)
    out = SAMGuidedMutatorMujoco._random_fallback(inp)
    assert isinstance(out, tuple)


# =============================================================================
# Part 5 — config builders (mock DI-engine / torch; optional EasyDict)
# =============================================================================


def _install_breakout_train_module_stubs() -> None:
    """Stubs so experiments_atari_sam.train_dreamerv3_breakout can import."""
    sys.modules.setdefault("torch", MagicMock())

    gym = types.ModuleType("gymnasium")

    class _Spaces:
        @staticmethod
        def Box(*args: Any, **kwargs: Any) -> object:
            return object()

        @staticmethod
        def Discrete(*args: Any) -> object:
            return object()

    gym.spaces = _Spaces()
    gym.Env = object
    gym.make = MagicMock()
    sys.modules["gymnasium"] = gym

    ding = types.ModuleType("ding")
    ding_envs = types.ModuleType("ding.envs")

    class BaseEnv:
        pass

    BaseEnvTimestep = object
    ding_envs.BaseEnv = BaseEnv
    ding_envs.BaseEnvTimestep = BaseEnvTimestep

    ding_torch_utils = types.ModuleType("ding.torch_utils")
    ding_torch_utils.to_ndarray = lambda x: np.asarray(x)

    ding_utils = types.ModuleType("ding.utils")

    class _Registry:
        def register(self, name: str):
            def deco(cls):
                return cls

            return deco

    ding_utils.ENV_REGISTRY = _Registry()

    sys.modules["ding"] = ding
    sys.modules["ding.envs"] = ding_envs
    sys.modules["ding.torch_utils"] = ding_torch_utils
    sys.modules["ding.utils"] = ding_utils


def test_breakout_fallback_config_structure() -> None:
    try:
        importlib.import_module("easydict")
    except ImportError:
        pytest.skip("easydict not installed")
    _install_breakout_train_module_stubs()
    tb = importlib.import_module("experiments_atari_sam.train_dreamerv3_breakout")
    cfg, _create = tb._build_breakout_fallback_config()
    assert cfg.world_model.model.obs_type == "RGB"
    assert cfg.world_model.model.action_size == 4
    assert cfg.world_model.model.action_type == "discrete"
    assert cfg.policy.model.actor_dist == "onehot"


def test_halfcheetah_config_structure() -> None:
    try:
        importlib.import_module("easydict")
    except ImportError:
        pytest.skip("easydict not installed")
    sys.modules.setdefault("torch", MagicMock())
    th = importlib.import_module("experiments_mujoco_sam.train_dreamerv3_halfcheetah")
    cfg, _create = th._build_halfcheetah_config()
    assert cfg.world_model.model.obs_type == "vector"
    assert cfg.world_model.model.state_size == 17
    assert cfg.world_model.model.action_size == 6
    assert cfg.world_model.model.action_type == "continuous"
    assert cfg.policy.model.actor_dist == "normal"
    assert cfg.env.env_id == "HalfCheetah-v5"


# 运行方式: cd /Users/jq/Documents/RL_Testing && pytest tests/test_sam_experiments.py -v
