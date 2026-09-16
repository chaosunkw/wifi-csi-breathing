#!/bin/zsh
set -eu
project_dir="${0:A:h}"
cd "$project_dir"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
  print 'The project virtual environment is missing. Follow README setup first.'
  exit 1
fi
exec "$project_dir/.venv/bin/python" -m wifi_csi_breathing gui --catalog "$project_dir/data/catalog.json"
