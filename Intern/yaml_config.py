import yaml
import re


def yaml_to_env(config_file: str) -> str:
    config = yaml.safe_load(config_file)
    env_list = []
    _recursive_yaml_to_env(config, env_list)
    pattern = r'\.(?==)'
    env_list_cleared = []
    for s in env_list:
        result = re.sub(pattern, '', s)
        env_list_cleared.append(result)
    return '\n'.join(env_list_cleared)


def _recursive_yaml_to_env(config, env_list, prefix=''):
    if isinstance(config, dict):
        for key, value in config.items():
            new_prefix = f"{prefix}{key}." #if prefix else key
            _recursive_yaml_to_env(value, env_list, new_prefix)
    else:
        if config is None:
            env_list.append(f"{prefix}")
        else:
            env_list.append(f"{prefix}={config}")


def env_to_yaml(env_list: str) -> str:
    config = {}
    for line in env_list.splitlines():
        if '=' in line:
            variable, value = line.strip().split('=')
            keys = variable.split('.')
            current = config
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = yaml.safe_load(value)
    return yaml.dump(config)