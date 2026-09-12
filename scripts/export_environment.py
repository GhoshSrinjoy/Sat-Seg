"""Record resolved versions after installation without pinning the input spec."""

import json
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs"
    destination.mkdir(exist_ok=True)
    export = subprocess.run(
        ["conda", "env", "export", "--name", "sat_clas"],
        capture_output=True, text=True, check=True,
    )
    environment = yaml.safe_load(export.stdout)
    environment.pop("prefix", None)
    # Avoid inheriting unrelated channels from the user's global conda config.
    source_spec = yaml.safe_load((root / "environment.yml").read_text(encoding="utf-8"))
    environment["channels"] = source_spec["channels"]
    for dependency in environment["dependencies"]:
        if isinstance(dependency, dict) and "pip" in dependency:
            # Reinstall this checkout separately; absolute editable paths are not portable.
            dependency["pip"] = [
                "--extra-index-url https://download.pytorch.org/whl/cu130",
                *[line for line in dependency["pip"]
                  if not line.startswith(("sat-clas==", "sat-clas @", "-e "))],
            ]
    header = "# Resolved Windows environment; generated after an unpinned installation.\n"
    (destination / "environment.resolved.yml").write_text(
        header + yaml.safe_dump(environment, sort_keys=False), encoding="utf-8"
    )
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True, text=True, check=True,
    )
    (destination / "installed_packages.json").write_text(
        json.dumps(json.loads(packages.stdout), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Recorded resolved environment and package inventory in {destination}")


if __name__ == "__main__":
    main()
