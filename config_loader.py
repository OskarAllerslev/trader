import os
import yaml

def get_config():
    """Load config.yaml from the config/ folder in the main repo."""
    # Always resolve relative to the location of config_loader.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "config.yaml")
    config_path = os.path.normpath(config_path)

    # Load the YAML file
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
