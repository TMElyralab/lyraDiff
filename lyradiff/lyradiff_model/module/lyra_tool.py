import yaml

def load_yaml(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)