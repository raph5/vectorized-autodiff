#include <stdio.h>
#include "../reverse.h"

int main() {
  tape_t *tape = tape_create(64);
  tape_load(tape);

  var_t a = var_create(4);
  var_t b = var_create(9);
  var_t c = var_create(7);
  var_t d = var_create(-2);

  var_t e = var_pow(var_sqrt(a / (b + c * a) + var_exp(var_create(1) / d)), -var_create(3));
  tape_reverse_pass(tape, e);
  printf("value: %f\n", var_value(e));
  printf("grad: {%f, %f, %f, %f}\n", var_adjoint(a), var_adjoint(b), var_adjoint(c), var_adjoint(d));

  tape_destroy(tape);
  return 0;
}
