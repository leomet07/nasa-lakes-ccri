#!/bin/bash

# script to remove all rapids ai stuff from requirements.txt so it's safe to install on macs (cpu only)


pip freeze > requirements.txt

sed -i.bak -r '/nvidia|cuml|cu12|cuda|dask|pynvml/d' requirements.txt