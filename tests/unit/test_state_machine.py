"""Unit tests for persistent state manager and phase transitions."""

from pathlib import Path

from hcs_image_evolution.agent.state_machine import EvolutionPhase, StateManager


def test_state_initialization_and_transition(tmp_path: Path):
    state_file = tmp_path / "run_state.json"
    mgr = StateManager(state_file=state_file)

    assert mgr.state.stage == EvolutionPhase.BOOT
    assert state_file.exists()

    mgr.transition_to(EvolutionPhase.DISCOVER)
    assert mgr.state.stage == EvolutionPhase.DISCOVER

    # Verify reloading from disk preserves state
    mgr2 = StateManager(state_file=state_file)
    assert mgr2.state.stage == EvolutionPhase.DISCOVER
    assert mgr2.state.run_id == mgr.state.run_id
