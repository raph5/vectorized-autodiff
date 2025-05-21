/*
 * ============================================================================
 * Simple Vectorized Forward Mode Autodiff
 * ============================================================================
 * This header-only C implementation provides forward mode automatic
 * differentiation using operator overloading on a custom `var_t` type.
 *
 * Each `var_t` variable holds:
 *  - `value`: the scalar value of the variable.
 *  - `grad[GRADLEN]`: a gradient vector representing the derivative of the
 *    variable with respect to each input in a vector of size `GRADLEN`.
 *
 * Usage Example:
 * ----------------------------------------------------------------------------
 * To compute ∂f/∂x and ∂f/∂y for f(x, y) = sin(x) + y²:
 *   var_t x = {.value = 1.0}; x.grad[0] = 1; // ∂x/∂x = 1
 *   var_t y = {.value = 2.0}; y.grad[1] = 1; // ∂y/∂y = 1
 *   var_t f = var_sin(x) + var_pow(y, 2);
 *   // f.value holds the result, f.grad[0] is ∂f/∂x, f.grad[1] is ∂f/∂y
 *
 * Notes:
 * ----------------------------------------------------------------------------
 *  - The macro `GRADLEN` must be defined before including this header.
 */

#ifndef H_AUTODIFF
#define H_AUTODIFF

#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

/* gradient length */
#ifndef GRADLEN
#error "The GRADLEN macro must set before including fowrard.h"
#define GRADLEN 0
#endif

typedef struct {
  float grad[GRADLEN];
  float value;
} var_t;

/*
 * initialize an new variable that does not derive from the input vector (see
 * above description)
 */
static void var_zero(var_t *a) {
  memset(a, 0, sizeof(*a));
}

/* variable operations */
static var_t operator-(var_t a) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = -a.grad[i];
  a.value = -a.value;
  return a;
}

/* variable variable operations */
static var_t operator+(var_t a, const var_t &b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = a.grad[i] + b.grad[i];
  a.value = a.value + b.value;
  return a;
}

static var_t operator-(var_t a, const var_t &b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = a.grad[i] - b.grad[i];
  a.value = a.value - b.value;
  return a;
}

static var_t operator*(var_t a, const var_t &b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = b.value * a.grad[i] + a.value * b.grad[i];
  a.value = a.value * b.value;
  return a;
}

static var_t operator/(var_t a, const var_t &b) {
  assert(b.value != 0);
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = (b.value * a.grad[i] - a.value * b.grad[i]) / (b.value * b.value);
  a.value = a.value / b.value;
  return a;
}

static void operator+=(var_t &a, const var_t &b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = a.grad[i] + b.grad[i];
  a.value = a.value + b.value;
}

static void operator-=(var_t &a, const var_t &b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = a.grad[i] - b.grad[i];
  a.value = a.value - b.value;
}

static void operator*=(var_t &a, const var_t &b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = b.value * a.grad[i] + a.value * b.grad[i];
  a.value = a.value * b.value;
}

static void operator/=(var_t &a, const var_t &b) {
  assert(b.value != 0);
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = (b.value * a.grad[i] - a.value * b.grad[i]) / (b.value * b.value);
  a.value = a.value / b.value;
}

/* variable float operations */
static var_t operator+(var_t a, float b) {
  a.value += b;
  return a;
}

static var_t operator-(var_t a, float b) {
  a.value -= b;
  return a;
}

static var_t operator*(var_t a, float b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] *= b;
  a.value *= b;
  return a;
}

static var_t operator/(float a, var_t b) {
  for (size_t i = 0; i < GRADLEN; i++)
    b.grad[i] = -a * b.grad[i] / (b.value * b.value);
  b.value = a / b.value;
  return b;
}

static void operator+=(var_t &a, float b) {
  a.value += b;
}

static void operator-=(var_t &a, float b) {
  a.value -= b;
}

static void operator*=(var_t &a, float b) {
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] *= b;
  a.value *= b;
}

static void operator/=(float a, var_t &b) {
  for (size_t i = 0; i < GRADLEN; i++)
    b.grad[i] = -a * b.grad[i] / (b.value * b.value);
  b.value = a / b.value;
}

/* variable functions */
static var_t var_pow(var_t a, float b) {
  assert(a.value > 0);
  float pow = powf(a.value, b-1);
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = b * a.grad[i] * pow;
  a.value = powf(a.value, b);
  return a;
}

static var_t var_exp(var_t a) {
  float expa = expf(a.value);
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = a.grad[i] * expa;
  a.value = expa;
  return a;
}

static var_t var_cos(var_t a) {
  float sina = -sinf(a.value);
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = a.grad[i] * sina;
  a.value = cosf(a.value);
  return a;
}

static var_t var_sin(var_t a) {
  float cosa = cosf(a.value);
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = a.grad[i] * cosa;
  a.value = sinf(a.value);
  return a;
}

static var_t var_sqrt(var_t a) {
  /* assert(a.value > 0); */
  for (size_t i = 0; i < GRADLEN; i++)
    a.grad[i] = 0.5 * a.grad[i] / sqrtf(a.value);
  a.value = sqrtf(a.value);
  return a;
}

#endif
