#include <stdio.h>

#define GRADLEN 4
#include "../forward.h"

int main() {
  var_t a = {.grad = {1, 0, 0, 0}, .value = 4};
  var_t b = {.grad = {0, 1, 0, 0}, .value = 9};
  var_t c = {.grad = {0, 0, 1, 0}, .value = 7};
  var_t d = {.grad = {0, 0, 0, 1}, .value = -2};

  var_t e = var_pow(var_sqrt(a / (b + c * a) + var_exp(1 / d)), -3);
  printf("value: %f\n", e.value);
  printf("grad: {%f, %f, %f, %f}\n", e.grad[0], e.grad[1], e.grad[2], e.grad[3]);

  return 0;
}
