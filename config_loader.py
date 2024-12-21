import os
import yaml

def get_config():
    """Load config.yaml dynamically based on the environment."""
    # Check if running locally (Windows path)
    if os.name == "nt":
        config_path = r"C:\Users\oskar\OneDrive\strategytrader\trader\config\config.yaml"
    else:
        # Default to container or relative path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.normpath(os.path.join(script_dir, "config", "config.yaml"))

    # Debugging output
    print(f"Resolved config path: {config_path}")

    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load the config file
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
