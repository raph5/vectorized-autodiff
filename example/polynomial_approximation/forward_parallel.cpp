#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <assert.h>

const int N = 1000;  /* number of terms in the reimann sum */
const int DEG = 10;  /* degree of the polynomial proximation */
const float START = 0;  /* the start of the integration interval */
const float END = 2;  /* the end of the integration interval */
const float ITERATIONS = 5000;  /* number of gradient descent iterations */
const float ALPHA = 0.001;  /* gradient descent speed */

#define GRADLEN 32
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

void poly_print(float P[DEG+1]) {
  printf("polynomial: ");
  for (size_t i = 0; i < DEG; ++i) {
    printf("%f, ", P[i]);
  }
  printf("%f\n", P[DEG]);
}

#define RI_WORKERS 2
const size_t RI_CHUNKS = ((DEG+1 + GRADLEN-1) / GRADLEN);

typedef struct {
  size_t start_chunk;
  size_t end_chunk;
  float *P;
  float *grad;
  float value;
} ri_worker_param_t;

void *ri_worker(void *param_ptr) {
  ri_worker_param_t *param = (ri_worker_param_t *) param_ptr;

  for (size_t chunk_id = param->start_chunk; chunk_id < param->end_chunk; ++chunk_id) {
    var_t loss = {0};
    var_t P[DEG+1] = {0};
    for (size_t i = 0; i < DEG+1; ++i) {
      P[i].value = param->P[i];
      if (i >= chunk_id * GRADLEN && i < chunk_id * GRADLEN + GRADLEN) {
        P[i].grad[i - chunk_id * GRADLEN] = 1;
      }
    }

    float step_size = (END-START) / N;
    for (size_t j = 0; j < N; ++j) {
      float x = START + j*step_size;
      var_t delta = poly_eval(P, x) - f(x);
      loss = loss + (delta*delta) * step_size;
      if (loss.value != loss.value) {

      }
    }

    if (chunk_id == param->start_chunk) {
      param->value = loss.value;
    } else {
      /* assert(param->value == loss.value); */
    }
    for (size_t i = 0; i < GRADLEN && chunk_id * GRADLEN + i < DEG+1; ++i) {
      param->grad[chunk_id * GRADLEN + i] = loss.grad[i];
    }
  }

  return NULL;
}

float reimann_integral(float P[DEG+1], float grad[DEG+1]) {
  pthread_t worker_threads[RI_WORKERS];
  ri_worker_param_t worker_params[RI_WORKERS];

  assert(RI_CHUNKS > 0);
  int handled_chunks = 0;
  for (size_t worker_id = 0; worker_id < RI_WORKERS; ++worker_id) {
    size_t start_chunk = (size_t) ((float) RI_CHUNKS / RI_WORKERS * worker_id);
    size_t end_chunk = (size_t) ((float) RI_CHUNKS / RI_WORKERS * (worker_id+1));
    worker_params[worker_id] = {
      .start_chunk = start_chunk,
      .end_chunk = end_chunk,
      .P = P,
      .grad = grad,
    };
    int err = pthread_create(&worker_threads[worker_id], NULL, &ri_worker, &worker_params[worker_id]);
    if (err) {
      printf("pthread_create error %d", err);
      exit(1);
      return 0;
    }
    handled_chunks += end_chunk - start_chunk;
  }
  assert(handled_chunks == RI_CHUNKS);

  for (size_t worker_id = 0; worker_id < RI_WORKERS; ++worker_id) {
    int err = pthread_join(worker_threads[worker_id], NULL);
    if (err) {
      printf("pthread_join error %d", err);
      exit(1);
      return err;
    }
  }

  float value;
  for (size_t worker_id = 1; worker_id < RI_WORKERS; ++worker_id) {
    if (worker_params[worker_id].end_chunk - worker_params[worker_id].start_chunk > 0) {
      value = worker_params[worker_id].value;
    }
  }
  return value;
}

void polynomial_approximation(float P[DEG+1]) {
  for (size_t i = 0; i < DEG+1; ++i) {
    P[i] = i+1;
  }

  for (size_t i = 0; i < ITERATIONS; ++i) {
    /* reimann integral */
    float loss_grad[DEG+1];
    float loss = reimann_integral(P, loss_grad);
    /* printf("loss: %f\n", loss); */

    /* gradient descent */
    for (size_t j = 0; j < DEG+1; ++j) {
      float one_over_norm_of_xj = j / (powf(END, j+1) - powf(START, j+1));
      /* normalize the influance of X^i */
      P[j] -= ALPHA * loss_grad[j] * one_over_norm_of_xj;
    }
  }
}

int main() {
  float P[DEG+1];
  polynomial_approximation(P);
  poly_print(P);

  return 0;
}
