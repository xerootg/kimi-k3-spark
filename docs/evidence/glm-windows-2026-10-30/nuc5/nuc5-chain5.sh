#!/bin/bash
cd /srv/glm-5.3-flash
until grep -q "SPARK-NUC5W-DONE" nuc5-win4.log; do sleep 5; done
sleep 3
bash spark-nuc5w.sh Bw Cw A2w C2w B2w PCw PBw Vw
