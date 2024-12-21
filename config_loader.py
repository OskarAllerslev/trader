import os
import yaml

def get_config():
    """Load config.yaml dynamically based on the current environment."""
    # Base path resolution for the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "config.yaml")

    # Debugging output
    print(f"Resolved config path: {config_path}")

    # Check if the config file exists
    if not os.path.exists(config_path):
        # If not found, provide a helpful error message
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load the config file
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
