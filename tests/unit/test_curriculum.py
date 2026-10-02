"""Unit tests for prompt curriculum and difficulty adaptation."""

from hcs_image_evolution.generation.prompt_curriculum import PromptCurriculum


def test_curriculum_sampling():
    curriculum = PromptCurriculum()
    bucket, prompt = curriculum.sample_prompt()
    assert bucket in curriculum.CATEGORIES
    assert isinstance(prompt, str)
    assert len(prompt) > 5


def test_curriculum_adaptation():
    curriculum = PromptCurriculum()
    initial_typo_weight = curriculum.weights["D5_typography"]
    curriculum.adapt_weights_for_weakness("typography", boost=0.50)
    assert curriculum.weights["D5_typography"] > initial_typo_weight
