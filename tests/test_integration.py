"""Integration tests: real Gymnasium pipelines for Atari Breakout & MuJoCo HalfCheetah (no DI-engine checkpoints)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "STARLA" / "src"))

import gymnasium as gym  # noqa: E402


def make_mock_agent(action: Any) -> MagicMock:
    """Simulate DreamerV3Agent for pipeline tests."""
    agent = MagicMock()
    agent.predict = MagicMock(return_value=(action, {}))
    agent.reset_state = MagicMock()
    agent.actor = MagicMock()
    agent.value_head = MagicMock()
    agent.device = MagicMock()
    return agent


def make_mock_world_model(obs_type: str = "RGB") -> MagicMock:
    wm = MagicMock()
    wm.obs_type = obs_type
    return wm


# =============================================================================
# Part 1 — Atari Breakout
# =============================================================================


def _require_ale_and_rom() -> None:
    pytest.importorskip("ale_py", reason="ale_py not installed")
    try:
        e = gym.make("ALE/Breakout-v5", obs_type="rgb")
        e.close()
    except Exception:
        pytest.skip("Breakout ROM not found, run: AutoROM --accept-license")


def _require_pil() -> None:
    """Pillow must import (used for Atari RGB resize in preprocess)."""
    try:
        import PIL  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Pillow not installed or not loadable: {exc}")


def test_atari_gym_make_and_step() -> None:
    _require_ale_and_rom()
    env = gym.make("ALE/Breakout-v5", obs_type="rgb")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (210, 160, 3)
        assert obs.dtype == np.uint8
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, _info = env.step(action)
        assert obs2.shape == (210, 160, 3)
        assert isinstance(reward, float)
    finally:
        env.close()


def test_atari_preprocess_ale_rgb_obs() -> None:
    _require_pil()
    _require_ale_and_rom()
    from experiments_atari_sam.train_dreamerv3_breakout import preprocess_ale_rgb_obs_for_dreamer

    raw_obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)
    out = preprocess_ale_rgb_obs_for_dreamer(raw_obs, image_size=64)
    assert out.shape == (3, 64, 64)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_atari_gymnasium_env_adapter() -> None:
    _require_ale_and_rom()
    from starla.envs.gymnasium_adapter import GymnasiumEnv

    env = GymnasiumEnv("ALE/Breakout-v5", obs_type="rgb")
    try:
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (210, 160, 3)
        action = 0
        obs2, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
    finally:
        env.env.close()


def test_atari_maybe_preprocess_with_obs_type_set() -> None:
    _require_pil()
    _require_ale_and_rom()
    from experiments_atari_sam.run_four_experiments_atari import _maybe_preprocess_obs_for_dreamer

    raw_obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)

    wm = make_mock_world_model(obs_type="RGB")
    out = _maybe_preprocess_obs_for_dreamer(raw_obs, wm)
    assert out.shape == (3, 64, 64)

    wm_no_attr = MagicMock(spec=[])
    out2 = _maybe_preprocess_obs_for_dreamer(raw_obs, wm_no_attr)
    assert out2.shape == (210, 160, 3)
    print("[WARNING] obs_type attribute missing on world_model — preprocessing skipped!")


def test_atari_breakout_ale_rgb_env() -> None:
    _require_pil()
    _require_ale_and_rom()
    from experiments_atari_sam.train_dreamerv3_breakout import BreakoutALERGBEnv

    env = BreakoutALERGBEnv({"env_id": "ALE/Breakout-v5", "image_size": 64})
    try:
        obs = env.reset()
        assert obs.shape == (3, 64, 64)
        assert obs.dtype == np.float32
        assert 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0
        timestep = env.step(0)
        assert timestep.obs.shape == (3, 64, 64)
    finally:
        env.close()


def test_atari_collect_one_episode_mock_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten Breakout episodes so the test finishes in reasonable time (full game can take many minutes)."""
    _require_pil()
    _require_ale_and_rom()
    from experiments_atari_sam.run_four_experiments_atari import _collect_agent_episodes
    from starla.envs.gymnasium_adapter import GymnasiumEnv as _GymnasiumEnvReal

    def _limited_gymnasium_env(env_id: str, **kwargs: Any) -> Any:
        kw = dict(kwargs)
        kw.setdefault("max_episode_steps", 128)
        return _GymnasiumEnvReal(env_id, **kw)

    monkeypatch.setattr(
        "experiments_atari_sam.run_four_experiments_atari.GymnasiumEnv",
        _limited_gymnasium_env,
    )

    mock_agent = make_mock_agent(action=np.array([0]))
    try:
        episodes = _collect_agent_episodes(
            mock_agent,
            "ALE/Breakout-v5",
            n=1,
            deterministic=True,
            seed=0,
            min_transitions=1,
            world_model=make_mock_world_model("RGB"),
        )
    except Exception as exc:
        pytest.fail(f"_collect_agent_episodes raised: {exc}")
    assert len(episodes) == 1
    assert episodes[0][-1][0] == "done"
    assert isinstance(episodes[0][-1][1], float)


# =============================================================================
# Part 2 — MuJoCo HalfCheetah
# =============================================================================


def _require_mujoco_env() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        e = gym.make("HalfCheetah-v5")
        e.close()
    except Exception:
        pytest.skip("HalfCheetah-v5 not available (install mujoco + gymnasium[mujoco])")


def test_mujoco_gym_make_and_step() -> None:
    _require_mujoco_env()
    env = gym.make("HalfCheetah-v5")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (17,)
        assert obs.dtype in (np.float64, np.float32)
        action = env.action_space.sample()
        assert action.shape == (6,)
        obs2, reward, terminated, truncated, _info = env.step(action)
        assert obs2.shape == (17,)
        assert isinstance(reward, float)
    finally:
        env.close()


def test_mujoco_normalize_action_continuous() -> None:
    from experiments_mujoco_sam.run_four_experiments_mujoco import _normalize_action

    action_6d = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    result = _normalize_action(action_6d)
    assert isinstance(result, np.ndarray)
    assert result.shape == (6,)
    assert not isinstance(result, (int, np.integer))


def test_mujoco_gymnasium_env_adapter_KNOWN_BUG() -> None:
    # BUG: gymnasium_adapter.py line 31 does `int(action)` which fails for
    # multi-dimensional continuous actions (HalfCheetah action_size=6).
    # Fix needed: change int(action) to action in gymnasium_adapter.py,
    # or wrap GymnasiumEnv for continuous action spaces.
    _require_mujoco_env()
    from starla.envs.gymnasium_adapter import GymnasiumEnv

    env = GymnasiumEnv("HalfCheetah-v5")
    try:
        env.reset()
        continuous_action = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        # NumPy 1.x vs 2.x wording differs ("size-1" vs "0-dimensional").
        with pytest.raises(TypeError, match=r"only .*arrays can be converted to Python scalars"):
            env.step(continuous_action)
    finally:
        env.env.close()


def test_mujoco_continuous_gymnasium_env_adapter_fixed() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from experiments_mujoco_sam.envs.continuous_gymnasium_adapter import ContinuousGymnasiumEnv

    env = ContinuousGymnasiumEnv("HalfCheetah-v5")
    try:
        env.reset()
        continuous_action = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
        obs, reward, terminated, truncated, info = env.step(continuous_action)
        assert obs.shape == (17,)
        assert isinstance(reward, float)
    finally:
        env.env.close()


def test_mujoco_gymnasium_env_adapter_discrete_still_works() -> None:
    from starla.envs.gymnasium_adapter import GymnasiumEnv

    env = GymnasiumEnv("CartPole-v1")
    try:
        env.reset()
        obs2, r, te, tr, info = env.step(0)
        assert isinstance(r, float)
    finally:
        env.env.close()


def test_mujoco_collect_random_episodes() -> None:
    _require_mujoco_env()
    from experiments_mujoco_sam.run_four_experiments_mujoco import _collect_random_episodes

    episodes = _collect_random_episodes("HalfCheetah-v5", n=2, seed=0)
    assert len(episodes) == 2
    for ep in episodes:
        assert ep[-1][0] == "done"
        assert isinstance(ep[-1][1], float)
    assert isinstance(episodes[0][0][0], np.ndarray)
    assert episodes[0][0][0].shape == (17,)
    assert isinstance(episodes[0][0][1], list)
    assert len(episodes[0][0][1]) == 6


def test_mujoco_collect_agent_episodes_KNOWN_BUG(monkeypatch: pytest.MonkeyPatch) -> None:
    # BUG: _collect_agent_episodes calls env.step(env_action) where env is
    # GymnasiumEnv. GymnasiumEnv.step() calls int(action) internally, crashing for 6D
    # continuous actions.
    _require_mujoco_env()
    from starla.envs.gymnasium_adapter import GymnasiumEnv as BrokenGymnasiumEnv

    monkeypatch.setattr(
        "experiments_mujoco_sam.run_four_experiments_mujoco.ContinuousGymnasiumEnv",
        BrokenGymnasiumEnv,
    )
    from experiments_mujoco_sam.run_four_experiments_mujoco import _collect_agent_episodes

    mock_agent = make_mock_agent(action=np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]))
    with pytest.raises(TypeError):
        _collect_agent_episodes(
            mock_agent,
            "HalfCheetah-v5",
            n=1,
            deterministic=True,
            seed=0,
            min_transitions=1,
        )


def test_mujoco_collect_agent_episodes_with_fix(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    from experiments_mujoco_sam.envs.continuous_gymnasium_adapter import ContinuousGymnasiumEnv
    from experiments_mujoco_sam.run_four_experiments_mujoco import _collect_agent_episodes

    def _short_halfcheetah(env_id: str, **kwargs: Any) -> Any:
        kw = dict(kwargs)
        kw.setdefault("max_episode_steps", 64)
        return ContinuousGymnasiumEnv(env_id, **kw)

    monkeypatch.setattr(
        "experiments_mujoco_sam.run_four_experiments_mujoco.ContinuousGymnasiumEnv",
        _short_halfcheetah,
    )

    mock_agent = make_mock_agent(action=np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]))
    episodes = _collect_agent_episodes(
        mock_agent,
        "HalfCheetah-v5",
        n=1,
        deterministic=True,
        seed=0,
        min_transitions=1,
    )
    assert len(episodes) == 1
    assert episodes[0][-1][0] == "done"
    assert isinstance(episodes[0][-1][1], float)


def test_atari_maybe_preprocess_nested_obs_type() -> None:
    pytest.importorskip("PIL", reason="Pillow not installed")
    from experiments_atari_sam.run_four_experiments_atari import _maybe_preprocess_obs_for_dreamer

    raw_obs = np.random.randint(0, 256, (210, 160, 3), dtype=np.uint8)

    class _NestedObsTypeWM:
        pass

    wm = _NestedObsTypeWM()
    wm.model = MagicMock()
    wm.model.obs_type = "RGB"

    out = _maybe_preprocess_obs_for_dreamer(raw_obs, wm)
    assert out.shape == (3, 64, 64), "nested obs_type should trigger preprocessing"


# 运行方式: cd /Users/jq/Documents/RL_Testing && pytest tests/test_integration.py -v
# Atari 需要: pip install ale-py shimmy Pillow && AutoROM --accept-license
# MuJoCo 需要: pip install mujoco gymnasium[mujoco]
