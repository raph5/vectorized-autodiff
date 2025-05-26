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

#ifndef GRADLEN
#warning "GRADLEN set to default value 8"
#define GRADLEN 8
#endif
#include "../../forward.h"

/* the function to approximate */
float f(float x) {
  if (x == 0) return 0;
  return exp(-1 / (x*x));
}

var_t poly_eval(var_t P[DEG+1], float x) {
  var_t val = P[0];
  float X = x;
  for (size_t i = 1; i < DEG+1; i++) {
    val += P[i] * X;
    X *= x;
  }
  return val;
}

void poly_init(var_t P[DEG+1], size_t grad_start, size_t grad_end) {
  for (size_t i = 0; i < DEG+1; ++i) {
    P[i] = {.grad = {0}, .value = (float) i+1};
    if (i >= grad_start && i < grad_end) {
      P[i].grad[i - grad_start] = 1;
    }
  }
}

var_t reimann_integral(var_t P[DEG+1]) {
  var_t loss = {0};

  float step_size = (END-START) / N;
  for (size_t j = 0; j < N; ++j) {
    float x = START + j*step_size;
    var_t delta = poly_eval(P, x) - f(x);
    loss = loss + (delta*delta) * step_size;
  }

  return loss;
}

int main() {
  size_t runs = 10;
  float start_time, end_time;
  var_t P[DEG+1];

  start_time = (float) clock() / CLOCKS_PER_SEC;
  for (size_t i = 0; i < runs; ++i) {
    for (size_t wrt_start = 0; wrt_start < DEG+1; wrt_start += GRADLEN) {
      poly_init(P, wrt_start, wrt_start + GRADLEN);
      var_t v = reimann_integral(P);
    }
  }
  end_time = (float) clock() / CLOCKS_PER_SEC;

  /* print average runtime in milliseconds */
  printf("%f", (end_time - start_time) / runs * 1000);
  return 0;
}
