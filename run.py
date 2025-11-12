import argparse
import importlib.util
import logging
import sys
from pathlib import Path

from icl.train import train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config_from_file(config_path: str):
    """Load config from a Python file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load the module from file
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)

    # Get the config
    if not hasattr(config_module, 'get_config'):
        raise AttributeError(f"Config file must have a 'get_config()' function")

    return config_module.get_config()


def main():
    parser = argparse.ArgumentParser(description='Train ICL model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., icl/configs/example.py)')
    args = parser.parse_args()

    logging.info("Starting training")
    logging.info(f"Loading config from: {args.config}")

    config = load_config_from_file(args.config)
    train(config)


if __name__ == "__main__":
    main()
