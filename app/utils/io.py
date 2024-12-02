import json
from pathlib import Path
from typing import Any, Dict

def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to save the JSON file
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(path: Path) -> Dict[str, Any]:
    """Load data from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary containing the loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)