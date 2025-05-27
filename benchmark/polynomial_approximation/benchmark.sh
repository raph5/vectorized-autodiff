#!/bin/bash

gradlen=64

bench() {
  reverse=$(./reverse_build_"$1")
  forward=$(./forward_build_"$1")
  forward_novec=$(./forward_build_novec_"$1")
  forward_gradlen=$(./forward_build_gradlen_"$1"_"$gradlen")
  echo "$1","$reverse","$forward","$forward_novec","$forward_gradlen"
}

deg=(1 2 $(seq 4 4 512))
for d in ${deg[@]}; do
  make -j build DEG=$d GRADLEN=$gradlen > /dev/null &
done
wait

for d in ${deg[@]}; do
  bench $d
done

make clean > /dev/null
