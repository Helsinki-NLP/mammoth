This directory contains some legacy scripts not deemed worthy to install as part of the mammoth package.

`check_wheel_commands.sh` installs a built wheel into a fresh venv and runs every `mammoth_*` command with `--help`. The release workflow runs it before publishing to PyPI.
