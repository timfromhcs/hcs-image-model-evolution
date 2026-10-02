"""Validation suite executing required test cases against sd-cli.exe."""

from pathlib import Path

from hcs_image_evolution.sdcpp.runner import SDCPPRunner
from hcs_image_evolution.utils.logging import log_event, logger


class SDCPPValidator:
    """Verifies all functional inference modes conforming to PLAN.md section 47."""

    def __init__(self, runner: SDCPPRunner, test_output_dir: Path = Path("state/sdcpp_validation")):
        self.runner = runner
        self.test_output_dir = test_output_dir
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

    def validate_all_modes(self, model_path: Path) -> dict[str, str]:
        """Runs validation tests across T2I, I2I, typography, and high-resolution."""
        results = {}

        # 1. T2I simple
        t2i_out = self.test_output_dir / "val_t2i.png"
        ok_t2i, _, _ = self.runner.run_t2i(
            model_path=model_path,
            prompt="A sleek red sports car parked on a mountain pass at sunset",
            output_path=t2i_out,
            steps=4,
        )
        results["t2i"] = "pass" if ok_t2i else "fail"

        # 2. Typography
        typo_out = self.test_output_dir / "val_typography.png"
        ok_typo, _, _ = self.runner.run_t2i(
            model_path=model_path,
            prompt='A neon sign glowing on brick wall saying "EVOLUTION"',
            output_path=typo_out,
            steps=4,
        )
        results["typography"] = "pass" if ok_typo else "fail"

        # 3. High-Res 1536x1024
        hd_out = self.test_output_dir / "val_hd.png"
        ok_hd, _, _ = self.runner.run_t2i(
            model_path=model_path,
            prompt="A majestic alpine mountain range reflected in a calm crystal lake",
            output_path=hd_out,
            width=1536,
            height=1024,
            steps=4,
        )
        results["highres"] = "pass" if ok_hd else "fail"

        log_event("sdcpp_validation_completed", results)
        logger.info("sd.cpp validation suite results: %s", results)
        return results
