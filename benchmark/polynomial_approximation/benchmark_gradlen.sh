#!/usr/bin/env bash

deg=300

bench() {
  forward_gradlen=$(./forward_build_gradlen_"$deg"_"$1")
  echo "$1","$forward_gradlen"
}

gradlen=($(seq 4 1 512))
for gl in ${gradlen[@]}; do
  make -j forward_gradlen DEG=$deg GRADLEN=$gl > /dev/null &
done
wait

for gl in ${gradlen[@]}; do
  bench $gl
done

make clean
