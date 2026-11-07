#!/bin/bash
# sm-run.sh "TAG BUILD MODE [ENV...]" ...  -> runs legs sequentially, each under its own flock
cd /srv/glm-5.3-flash
for spec in "$@"; do flock window.lock ./sm-leg3.sh $spec; sleep 5; done
echo ALL-DONE
