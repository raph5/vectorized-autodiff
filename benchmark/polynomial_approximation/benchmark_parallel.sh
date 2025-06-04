#!/bin/bash

bench() {
  parallel=$(./parallel_build_"$1")
  echo "$d","$parallel"
}

deg=(4 8 $(seq 4 16 512))
for d in ${deg[@]}; do
  make -j parallel DEG=$d > /dev/null &
done
wait

for d in ${deg[@]}; do
  bench $d
done

make clean
