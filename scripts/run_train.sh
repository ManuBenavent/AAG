#!/bin/bash

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <config_file>"
  exit 1
fi

# Check if the config file exists
if [ -f "$1" ]; then
  config=$1
else
  echo "Config file '$1' not found"
  exit 1
fi

name=$(python3 -c "import yaml;print(yaml.full_load(open('${config}'))['name'])")
dataset=$(python3 -c "import yaml;print(yaml.full_load(open('${config}'))['data']['dataset'])")
now=$(date +"%Y%m%d_%H%M%S")
mkdir -p exp/${dataset}/${name}/${now}
python3 -u main.py  --config ${config} --log_time $now 2>&1|tee exp/${dataset}/${name}/${now}/${now}.log