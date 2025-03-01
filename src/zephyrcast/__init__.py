import tomlkit

with open("pyproject.toml", "r") as f:
    toml_config = tomlkit.parse(f.read())
project_config = toml_config["tool"]["zephyrcast"]
