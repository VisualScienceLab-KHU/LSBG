import argparse

from .config import PROJECT_ROOT, configure_environment, load_config


def main(mode):
    parser = argparse.ArgumentParser(
        description=f"ReGFLoW {mode}: all experiment settings come from JSON."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / f"{mode}.json"),
        help="JSON configuration file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration, required files and data contracts without running the model",
    )
    args = parser.parse_args()
    config = load_config(args.config, expected_mode=mode)
    configure_environment(config)
    from . import workflows
    from .runtime import launch

    workflows.check_inputs(config, inspect_data=args.check)
    if args.check:
        return
    launch(getattr(workflows, mode), config)
