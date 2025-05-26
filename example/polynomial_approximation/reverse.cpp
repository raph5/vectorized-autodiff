#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <math.h>
#include <time.h>

const int N = 1000;  /* number of terms in the reimann sum */
const int DEG = 10;  /* degree of the polynomial proximation */
const float START = 0;  /* the start of the integration interval */
const float END = 2;  /* the end of the integration interval */
const float ITERATIONS = 5000;  /* number of gradient descent iterations */
const float ALPHA = 0.001;  /* gradient descent speed */

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
    val += P[i] * var_create(X);
    X *= x;
  }
  return val;
}

void poly_init(var_t P[DEG+1]) {
  for (size_t i = 0; i < DEG+1; ++i) {
    P[i] = var_create(i+1);
  }
}

void poly_print(float P[DEG+1]) {
  printf("polynomial: ");
  for (size_t i = 0; i < DEG; ++i) {
    printf("%f, ", P[i]);
  }
  printf("%f\n", P[DEG]);
}

var_t reimann_integral(var_t P[DEG+1]) {
  var_t loss = {0};

  float step_size = (END-START) / N;
  for (size_t j = 0; j < N; ++j) {
    float x = START + j*step_size;
    var_t delta = poly_eval(P, x) - var_create(f(x));
    loss = loss + (delta*delta) * var_create(step_size);
  }

  return loss;
}

void polynomial_approximation(float P_coef[DEG+1]) {
  tape_t *tape = tape_create(64);
  tape_load(tape);

  var_t P[DEG+1];
  poly_init(P);

  for (size_t i = 0; i < ITERATIONS; ++i) {
    /* reimann integral */
    var_t loss = reimann_integral(P);
    /* printf("loss: %f\n", var_value(loss)); */

    /* gradient descent */
    tape_reverse_pass(tape, loss);
    for (size_t j = 0; j < DEG+1; ++j) {
      /* normalize the influance of X^j */
      float one_over_norm_of_xj = j / (powf(END, j+1) - powf(START, j+1));
      P_coef[j] = var_value(P[j]) - ALPHA * var_adjoint(P[j]) * one_over_norm_of_xj;
    }

    /* update polynomial */
    tape_clear(tape);
    for (size_t j = 0; j < DEG+1; ++j) {
      P[j] = var_create(P_coef[j]);
    }
  }

  tape_destroy(tape);
}

int main() {
  float P[DEG+1];
  polynomial_approximation(P);
  poly_print(P);

  return 0;
}
