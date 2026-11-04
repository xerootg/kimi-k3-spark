#!/bin/bash
# runs legs in order; on server failure retries with a smaller HBM tier
cd /srv/glm-5.3
for U in 2048 4096 8192; do
  for H in 16384 8192 0; do
    tag=u${U}-h${H}
    bash pfub/leg.sh $tag $U $H > pfub/leg-$tag.out 2>&1
    cat pfub/leg-$tag.out
    grep -q -E "SERVER-FAIL|CLIENT-FAIL" pfub/leg-$tag.out || break
  done
done
echo ALL-DONE
