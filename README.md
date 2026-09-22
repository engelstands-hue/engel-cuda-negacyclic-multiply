<!-- GENERATED from polymul/negacyclic@1.0.0. Do not edit — ``--update`` rewrites it. -->
# Negacyclic ring, wide prime modulus

An answer to `polymul/negacyclic@1.0.0`, in cpp.

## What you write

`solve.cu`, and nothing else.

```
init(Point)         -> state     setup. Not measured.
run(state, Inputs)  -> Outputs   the task. Measured, and only this.
free(state)                      optional.
```

`init` never sees the data, so there is nothing for it to compute ahead of
time. Reading the case and writing the answer happen outside the measured
call, in generated code — which is why seeing the inputs early is not a way
to answer early.

## The shapes

- `a` — tensor<N x L x u32>
- `b` — tensor<N x L x u32>
- returns `c` — tensor<N x L x u32>

## Generated, and not yours

Every file but the one above carries a header saying so. They are rewritten
by `fherma implementation init --update`, which never touches yours.
