"""Curriculum-driven multi-dimensional prompt generator with adaptive difficulty weighting."""

import random


class PromptCurriculum:
    """Generates structured generation prompts categorized across difficulty levels D0 to D9."""

    CATEGORIES = {
        "D0_simple": [
            "A studio portrait of a golden retriever with soft studio lighting",
            "A ripe red apple sitting on a rustic wooden table",
            "A vintage mechanical pocket watch on velvet fabric",
        ],
        "D1_multiple_objects": [
            "A blue ceramic coffee mug next to a stack of leatherbound books and eyeglasses",
            "Three different types of citrus fruits: a sliced lemon, a whole lime, and a peeled orange",
            "A stainless steel thermos and a pair of hiking boots beside a backpack",
        ],
        "D2_detailed_scene": [
            "An intricate cyberpunk street food stall in Tokyo at night with glowing neon reflections on wet asphalt",
            "A Victorian greenhouse interior overflowing with exotic ferns, orchids, and sunlight filtering through glass panes",
            "A bustling artisan blacksmith workshop with glowing embers, hanging iron tools, and flying sparks",
        ],
        "D3_spatial_relationships": [
            "A small glass marble directly inside a wooden box, with a bronze key resting to the left of the box",
            "A cat sitting on top of an old TV monitor while a dog sleeps underneath the wooden desk",
            "A tall lighthouse on a cliff overlooking a sailboat sailing to the right of the peninsula",
        ],
        "D4_counting": [
            "Exactly five red billiard balls arranged neatly in a straight line on green felt",
            "A white bowl containing precisely three green apples and four yellow bananas",
            "Seven candles of varying heights burning brightly on a stone fireplace mantle",
        ],
        "D5_typography": [
            'A modern cafe storefront with a clean black awning that says "HCS ROASTERY" in bold white sans-serif lettering',
            'A vintage travel poster of the Swiss Alps featuring the prominent headline "EXPLORE THE HEIGHTS"',
            'A neon sign hanging on a brick wall glowing in bright turquoise with the word "EVOLUTION"',
        ],
        "D6_complex_composition": [
            "A high-angle bird's eye view of a Venetian canal crowded with gondolas, stone bridges, and architectural facades",
            "A split-level underwater shot showing coral reef life below and a sailboat with sunset above the waterline",
            "A multi-layered collage of architectural blueprints, copper gears, and botanical illustrations in a steampunk style",
        ],
        "D7_multiref_editing": [
            "Change the color of the car from red to matte midnight blue while preserving all metallic reflections",
            "Replace the summer trees in the background with snow-covered winter pines, keeping the cabin unchanged",
            "Add a pair of classic aviator sunglasses onto the portrait subject without altering facial geometry",
        ],
        "D8_realworld_editing": [
            "Seamlessly remove the person in the yellow jacket on the left side of the beach and reconstruct the ocean background",
            "Change the daylight scene to golden hour lighting with long warm shadows across the grass",
            "Alter the texture of the modern sofa from smooth fabric to worn brown chesterfield leather",
        ],
        "D9_adversarial": [
            "A transparent glass cube floating in zero gravity reflecting a checkerboard room with refracted caustics",
            "A mirror reflecting an impossible room where gravity pulls sideways, photorealistic optical distortion",
            "Extremely high-density flock of starlings forming an intricate murmuration over a calm mirror lake at twilight",
        ],
    }

    def __init__(self, difficulty_weights: dict[str, float] | None = None):
        self.weights = difficulty_weights or {
            "D0_simple": 0.10,
            "D1_multiple_objects": 0.15,
            "D2_detailed_scene": 0.15,
            "D3_spatial_relationships": 0.10,
            "D4_counting": 0.10,
            "D5_typography": 0.15,
            "D6_complex_composition": 0.10,
            "D7_multiref_editing": 0.05,
            "D8_realworld_editing": 0.05,
            "D9_adversarial": 0.05,
        }

    def adapt_weights_for_weakness(self, weakness: str, boost: float = 0.20) -> None:
        """Dynamically increases the proportion of prompts targeting identified benchmark weaknesses."""
        key_map = {
            "typography": "D5_typography",
            "counting": "D4_counting",
            "composition": "D6_complex_composition",
            "editing": "D7_multiref_editing",
        }
        target_key = key_map.get(weakness.lower())
        if target_key and target_key in self.weights:
            self.weights[target_key] += boost
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

    def sample_prompt(self) -> tuple[str, str]:
        """Samples a prompt and returns (difficulty_bucket, prompt_string)."""
        keys = list(self.weights.keys())
        probabilities = [self.weights[k] for k in keys]
        chosen_cat = random.choices(keys, weights=probabilities, k=1)[0]
        prompts = self.CATEGORIES[chosen_cat]
        return chosen_cat, random.choice(prompts)
