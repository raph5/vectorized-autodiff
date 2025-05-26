#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <math.h>
#include <time.h>

const int N = 1000;  /* number of terms in the reimann sum */
const float START = 0;  /* the start of the integration interval */
const float END = 2;  /* the end of the integration interval */
#ifndef DEG
#warning "DEG set to default value 4"
const int DEG = 4;  /* degree of the polynomial proximation */
#endif

#include "../../reverse.h"

/* the function to approximate */
float f(float x) {
  if (x == 0) return 0;
  return exp(-1 / (x*x));
}

var_t poly_eval(var_t P[DEG+1], float x) {
  var_t val = P[0];
  float X = x;
  for (size_t i = 1; i < DEG+1; i++) {
    val = val + P[i] * var_create(X);
    X *= x;
  }
  return val;
}

void poly_init(var_t P[DEG+1]) {
  for (size_t i = 0; i < DEG+1; ++i) {
    P[i] = var_create(i+1);
  }
}

var_t reimann_integral(var_t P[DEG+1]) {
  var_t loss = var_create(0);

  float step_size = (END-START)/N;
  for (size_t j = 0; j < N; ++j) {
    float x = START + j*step_size;
    var_t delta = poly_eval(P, x) - var_create(f(x));
    loss = loss + (delta*delta) * var_create(step_size);
  }

  return loss;
}

int main() {
  size_t runs = 10;
  float start_time, end_time;

  start_time = (float) clock() / CLOCKS_PER_SEC;
  for (size_t i = 0; i < 10; ++i) {
    var_t P[DEG+1];
    tape_t tape = tape_create(64);
    tape_load(tape);
    poly_init(P);
    var_t loss = reimann_integral(P);
    tape_destroy(tape);
  }
  end_time = (float) clock() / CLOCKS_PER_SEC;

  /* print average runtime in milliseconds */
  printf("%f", (end_time - start_time) / runs * 1000);
  return 0;
}
