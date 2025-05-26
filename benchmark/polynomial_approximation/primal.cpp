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

/* the function to approximate */
float f(float x) {
  if (x == 0) return 0;
  return exp(-1 / (x*x));
}

float poly_eval(float P[DEG+1], float x) {
  float val = P[0];
  float X = x;
  for (size_t i = 1; i < DEG+1; i++) {
    val = val + P[i] * X;
    X *= x;
  }
  return val;
}

void poly_init(float P[DEG+1]) {
  for (size_t i = 0; i < DEG+1; ++i) {
    P[i] = i+1;
  }
}

float reimann_integral(float P[DEG+1]) {
  float loss = 0;

  float step_size = (END-START)/N;
  for (size_t j = 0; j < N; ++j) {
    float x = START + j*step_size;
    float delta = poly_eval(P, x) - f(x);
    loss = loss + (delta*delta) * step_size;
  }

  return loss;
}

int main() {
  size_t runs = 10;
  float start_time, end_time;
  float P[DEG+1];

  start_time = (float) clock() / CLOCKS_PER_SEC;
  for (size_t i = 0; i < runs; ++i) {
    poly_init(P);
    reimann_integral(P);
  }
  end_time = (float) clock() / CLOCKS_PER_SEC;

  /* print average runtime in milliseconds */
  printf("%f", (end_time - start_time) / runs * 1000);
  return 0;
}
