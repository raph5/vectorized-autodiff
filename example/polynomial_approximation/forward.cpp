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

#define GRADLEN (DEG+1)
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
    var_t delta = poly_eval(P, x) - f(x);
    loss = loss + (delta*delta) * step_size;
  }

  return loss;
}

void polynomial_approximation(float P_coef[DEG+1]) {
  var_t P[DEG+1];
  poly_init(P, 0, GRADLEN);

  for (size_t i = 0; i < ITERATIONS; ++i) {
    /* reimann integral */
    var_t loss = reimann_integral(P);
    /* printf("loss: %f\n", loss.value); */

    /* gradient descent */
    for (size_t j = 0; j < DEG+1; ++j) {
      float one_over_norm_of_xj = j / (powf(END, j+1) - powf(START, j+1));
      /* normalize the influance of X^i */
      P[j].value -= ALPHA * loss.grad[j] * one_over_norm_of_xj;
    }
  }

  for (size_t i = 0; i < DEG+1; ++i) {
    P_coef[i] = P[i].value;
  }
}

int main() {
  float P[DEG+1];
  polynomial_approximation(P);
  poly_print(P);

  return 0;
}
