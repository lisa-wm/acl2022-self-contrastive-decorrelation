#! /bin/sh

rsync -auv \
--exclude .git \
--exclude .json \
--exclude .venv \
--exclude .data \
--exclude .idea \
--exclude .mypy_cache \
--exclude results_* \
--exclude models \
--exclude logs \
--max-size=50m \
. \
wimmerl@gpuserver.stat.uni-muenchen.de:projects/scd_multihead