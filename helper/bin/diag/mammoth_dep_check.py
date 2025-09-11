#!/usr/bin/env python3

import pkg_resources
import importlib.util
import os
import re
from pathlib import Path

# List of required packages with optional version constraints
required_packages = {
    "configargparse": None,
    "einops": "==0.8.0",
    "flake8": "==4.0.1",
    "flask": "==2.0.3",
    "pyonmttok": ">=1.32,<2",
    "pytest-flake8": "==1.1.1",
    "pytest": "==7.0.1",
    "pyyaml": None,
    "sacrebleu": "==2.3.1",
    "scikit-learn": "==1.2.0",
    "sentencepiece": "==0.1.97",
    "tensorboard": ">=2.9",
    "timeout-decorator": None,
    "torch": ">=1.10.2",
    "tqdm": "==4.66.2",
    "waitress": None,
    "x-transformers": "==1.32.14"
}

# Map package names to import names if different
import_aliases = {
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "pytest-flake8": "pytest_flake8",
    "x-transformers": "x_transformers"
}

# Parse logs to get the earliest valid (timestamp, module) for each package version match
log_dir = Path.home() / ".module_load_logs"
first_valid_seen_map = {}  # pkg_name -> (timestamp, module_name, version)

# Normalize PyPI names consistently for matching (dashes instead of underscores)
def normalize_pkg_name(name):
    return name.replace("_", "-").lower()

# Parse logs to get the earliest valid (timestamp, module) for each package version match
if log_dir.exists():
    log_files = sorted(log_dir.glob("*.*"))
    for log_file in log_files:
        match = re.match(r"(\d{8}-\d{6})_(.+?)\.(txt|log)$", log_file.name)
        if not match:
            continue
        timestamp, raw_module_name, _ = match.groups()
        module_name = normalize_pkg_name(raw_module_name)
        try:
            with open(log_file) as f:
                lines = f.readlines()
            for line in lines:
                if "==" not in line:
                    continue
                parts = line.strip().split("==")
                if len(parts) != 2:
                    continue
                raw_pkg_name = parts[0].strip()
                version_str = parts[1].strip()
                pkg_name = normalize_pkg_name(raw_pkg_name)

                if pkg_name in map(normalize_pkg_name, required_packages.keys()) and pkg_name not in first_valid_seen_map:
                    constraint = required_packages.get(pkg_name) or required_packages.get(raw_pkg_name)
                    try:
                        if not constraint or pkg_resources.Requirement.parse(f"{pkg_name}{constraint}").specifier.contains(version_str):
                            first_valid_seen_map[pkg_name] = (timestamp, module_name, version_str)
                    except Exception:
                        continue
        except Exception as e:
            print(f"⚠️ Failed to read log file {log_file.name}: {e}")

# Output the report

# Output the report
print("🔍 Checking MAMMOTH requirements...\n")
print("| **Status** | **Package**         | **Version**        | **Location** | **Loaded** |")
print("|------------|---------------------|--------------------|--------------|------------|")

home_path = str(Path.home())

for pkg, constraint in required_packages.items():  
    import_name = import_aliases.get(pkg, pkg).replace("-", "_")
    # import_name = import_aliases.get(pkg, pkg)
    try:
        dist = pkg_resources.get_distribution(pkg)
        version = dist.version
        status = "✅"

        # Check version constraint
        try:
            if constraint:
                pkg_resources.require(f"{pkg}{constraint}")
        except pkg_resources.VersionConflict:
            status = "⚠️  "

        # Get install location
        spec = importlib.util.find_spec(import_name)
        location = spec.origin if spec and spec.origin else "(unknown)"

        if location:
            # Replace home directory path with ~
            location = location.replace(home_path, "~")
            # Remove trailing '__init__.py' if present
            if location.endswith("__init__.py"):
                location = os.path.dirname(location) + "/."

        # Append first valid seen info if available
        seen_info = first_valid_seen_map.get(pkg.lower())
        if seen_info:
            timestamp, module_name, seen_version = seen_info
            location += f" | {module_name}"

        print(f"| {status:10} | {pkg:19} | {version:18} {'(ok)' if status == '✅' else f'(not {constraint})'} | `{location}` |")

    except pkg_resources.DistributionNotFound:
        print(f"| ❌         | {pkg:19} | (not installed)   | - |")
    except Exception as e:
        print(f"| ❌         | {pkg:19} | Error: {str(e)}   | - |")

print("\n✅ Done.")
