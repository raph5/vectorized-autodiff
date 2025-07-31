#!/usr/bin/env bash

d=500

bench() {
  parallel=$(./parallel_build_workers_"$d"_"$1")
  echo "$1","$parallel"
}

workers=$(seq 1 1 12)
for w in ${workers[@]}; do
  make -j parallel_workers DEG=$d WORKERS=$w > /dev/null &
done
wait

for w in ${workers[@]}; do
  bench $w
done

make clean
