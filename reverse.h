/*
 * ============================================================================
 * Simple Tape-Based Reverse Mode Autodiff
 * ============================================================================
 * This header-only C implementation provides reverse mode automatic
 * differentiation using a dynamic computation tape and operator overloading
 * on a custom `var_t` type.
 *
 * Each `var_t` variable corresponds to a node on a global tape. The tape
 * records the computation graph by tracking the operation and parent variables
 * for each intermediate result.
 *
 * After constructing a function from input `var_t`s, a reverse pass propagates
 * gradients from the output node backward through the tape using the chain
 * rule.
 *
 * Usage Example:
 * ----------------------------------------------------------------------------
 * To compute ∂f/∂x and ∂f/∂y for f(x, y) = sin(x) + y²:
 *   tape_t tape = tape_create(64);
 *   tape_load(tape);
 *   var_t x = var_create(1.0f);
 *   var_t y = var_create(2.0f);
 *   var_t f = var_sin(x) + var_pow(y, var_create(2.0f));
 *   tape_reverse_pass(tape, f);
 *   // var_adjoint(x) returns ∂f/∂x, var_adjoint(y) returns ∂f/∂y
 *
 * Notes:
 * ----------------------------------------------------------------------------
 *  - Always call `tape_load()` before creating variables.
 */

#ifndef H_AUTODIFF
#define H_AUTODIFF

#include <stddef.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>

const uint32_t MAX_TAPE_LENGTH = 1 << 24;  /* correspond to a ~330mb tap */

typedef enum {
  NIL = 0,
  NEG,
  ADD,
  SUB,
  MUL,
  DIV,
  POW,
  EXP,
  COS,
  SIN,
  SQRT,
} operator_t;

typedef struct {
  float value;
  float adjoint;
  uint32_t left_parent;
  uint32_t right_parent;
  operator_t op;
} tape_entry_t;

typedef struct {
  uint32_t length;
  uint32_t capacity;
  tape_entry_t *entries;
} tape_t;

typedef struct {
  uint32_t index;
} var_t;

/* should not be set directly, use `tape_load` instead */
static tape_t global_tape = {
  .length = 0,
  .capacity = 0,
  .entries = NULL,
};

/*
 * setting the initial capacity of the tape to a number like 64 will prevent too
 * much calls to realloc
 */
static tape_t tape_create(uint32_t capacity) {
  assert(capacity <= MAX_TAPE_LENGTH);
  tape_entry_t *entries = (tape_entry_t *) calloc(capacity, sizeof(tape_entry_t));
  if (entries == NULL) {
    perror("tape malloc");
    exit(1);
    return {};
  }
  return {
    .length = 0,
    .capacity = capacity,
    .entries = entries,
  };
}

static void tape_destroy(tape_t tape) {
  free(tape.entries);
}

static void tape_extend(tape_t tape) {
  assert(tape.length < MAX_TAPE_LENGTH);
  if (tape.length == tape.capacity) {
    tape.entries = (tape_entry_t *) realloc(tape.entries, 2 * tape.capacity * sizeof(*tape.entries));
    if (tape.entries == NULL) {
      perror("tape realloc");
      exit(1);
      return;
    }
    memset(tape.entries + tape.capacity, 0, tape.capacity * sizeof(*tape.entries));
    tape.capacity = 2 * tape.capacity;
  }
  ++tape.length;
}

static void tape_clear(tape_t tape) {
  memset(tape.entries, 0, tape.length * sizeof(*tape.entries));
  tape.length = 0;
}

static void tape_load(tape_t tape) {
  global_tape = tape;
}

static tape_t tape_loaded() {
  return global_tape;
}

static void tape_reverse_pass(tape_t tape, var_t start) {
  for (size_t i = 0; i < tape.length; ++i)
    tape.entries[i].adjoint = 0;
  tape.entries[start.index].adjoint = 1;

  for (size_t i = start.index+1; i-- > 0;) {  /* avoid size_t wraps */
    tape_entry_t *entry = &tape.entries[i];
    tape_entry_t *left_parent_entry = &tape.entries[entry->left_parent];
    tape_entry_t *right_parent_entry = &tape.entries[entry->right_parent];
    switch (entry->op) {
      case NIL:
        break;
      case NEG:
        left_parent_entry->adjoint += entry->adjoint * -1;
        break;
      case ADD:
        left_parent_entry->adjoint  += entry->adjoint * 1;
        right_parent_entry->adjoint += entry->adjoint * 1;
        break;
      case SUB:
        left_parent_entry->adjoint  += entry->adjoint * 1;
        right_parent_entry->adjoint += entry->adjoint * -1;
        break;
      case MUL:
        left_parent_entry->adjoint  += entry->adjoint * right_parent_entry->value;
        right_parent_entry->adjoint += entry->adjoint * left_parent_entry->value;
        break;
      case DIV:
        left_parent_entry->adjoint  += entry->adjoint / right_parent_entry->value;
        right_parent_entry->adjoint += entry->adjoint * -1 * (entry->value / right_parent_entry->value);
        break;
      case POW:
        left_parent_entry->adjoint  += entry->adjoint * right_parent_entry->value * (entry->value / left_parent_entry->value);
        right_parent_entry->adjoint += entry->adjoint * entry->value * logf(left_parent_entry->value);
        break;
      case EXP:
        left_parent_entry->adjoint += entry->adjoint * entry->value;
        break;
      case COS:
        left_parent_entry->adjoint += entry->adjoint * -1 * sqrtf(1 - entry->value*entry->value);
        break;
      case SIN:
        left_parent_entry->adjoint += entry->adjoint * sqrtf(1 - entry->value*entry->value);
        break;
      case SQRT:
        left_parent_entry->adjoint += entry->adjoint / (2 * entry->value);
        break;
    }
  }
}

/* append new variable to global_tape */
static var_t var_create(float value) {
  var_t a = {global_tape.length};
  tape_extend(global_tape);
  global_tape.entries[a.index].value = value;
  return a;
}

static float var_adjoint(var_t a) {
  return global_tape.entries[a.index].adjoint;
}

static float var_value(var_t a) {
  return global_tape.entries[a.index].value;
}

/* variable operations */
static var_t operator-(var_t a) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  var_t b = var_create(-a_entry->value);
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  b_entry->op = NEG;
  b_entry->left_parent = a.index;
  return b;
}


/* variable variable operations */
static var_t operator+(var_t a, var_t b) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  var_t c = var_create(a_entry->value + b_entry->value);
  tape_entry_t *c_entry = &global_tape.entries[c.index];
  c_entry->op = ADD;
  c_entry->left_parent = a.index;
  c_entry->right_parent = b.index;
  return c;
}

static var_t operator-(var_t a, var_t b) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  var_t c = var_create(a_entry->value - b_entry->value);
  tape_entry_t *c_entry = &global_tape.entries[c.index];
  c_entry->op = SUB;
  c_entry->left_parent = a.index;
  c_entry->right_parent = b.index;
  return c;
}

static var_t operator*(var_t a, var_t b) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  var_t c = var_create(a_entry->value * b_entry->value);
  tape_entry_t *c_entry = &global_tape.entries[c.index];
  c_entry->op = MUL;
  c_entry->left_parent = a.index;
  c_entry->right_parent = b.index;
  return c;
}

static var_t operator/(var_t a, var_t b) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  var_t c = var_create(a_entry->value / b_entry->value);
  assert(b_entry->value != 0);
  tape_entry_t *c_entry = &global_tape.entries[c.index];
  c_entry->op = DIV;
  c_entry->left_parent = a.index;
  c_entry->right_parent = b.index;
  return c;
}

static void operator+=(var_t &a, var_t b) {
  a = a + b;
}

static void operator-=(var_t &a, var_t b) {
  a = a - b;
}

static void operator*=(var_t &a, var_t b) {
  a = a * b;
}

static void operator/=(var_t &a, var_t b) {
  a = a / b;
}

/* variable functions */
static var_t var_pow(var_t a, var_t b) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  assert(a_entry->value > 0);
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  var_t c = var_create(powf(a_entry->value, b_entry->value));
  tape_entry_t *c_entry = &global_tape.entries[c.index];
  c_entry->op = POW;
  c_entry->left_parent = a.index;
  c_entry->right_parent = b.index;
  return c;
}

static var_t var_exp(var_t a) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  var_t b = var_create(expf(a_entry->value));
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  b_entry->op = EXP;
  b_entry->left_parent = a.index;
  return b;
}

static var_t var_cos(var_t a) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  var_t b = var_create(cosf(a_entry->value));
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  b_entry->op = COS;
  b_entry->left_parent = a.index;
  return b;
}

static var_t var_sin(var_t a) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  var_t b = var_create(sinf(a_entry->value));
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  b_entry->op = SIN;
  b_entry->left_parent = a.index;
  return b;
}

static var_t var_sqrt(var_t a) {
  tape_entry_t *a_entry = &global_tape.entries[a.index];
  /* assert(a_entry->value > 0); */
  var_t b = var_create(sqrtf(a_entry->value));
  tape_entry_t *b_entry = &global_tape.entries[b.index];
  b_entry->op = SQRT;
  b_entry->left_parent = a.index;
  return b;
}

#endif
