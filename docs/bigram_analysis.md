# Bigram Analysis: Optimal Letter Groupings

## Why grouping matters

In BESE, 18 less-common letters are encoded as 2 tokens: a group token + a position token. Letters sharing a group are ambiguous until the position token resolves them. To minimize prediction difficulty, we want letters in the same group to appear in *different* contexts, so the model can distinguish them from surrounding tokens.

## Context profiles

Each letter has a "context profile": which letters typically precede and follow it.

```
h: preceded by [t(100), c(23), w(20)]   followed by [e(93), a(48), i(44)]
l: preceded by [a(38), e(25), i(23)]     followed by [e(39), i(31), a(24)]
d: preceded by [n(56), e(49), a(16)]     followed by [e(32), i(16), o(13)]
c: preceded by [i(27), a(16), e(12)]     followed by [o(28), e(28), h(23)]
u: preceded by [o(50), s(15), q(14)]     followed by [s(24), r(22), t(21)]
m: preceded by [o(21), e(17), a(12)]     followed by [e(29), a(22), o(13)]
f: preceded by [o(33), l(8)]             followed by [o(20), e(13), i(13)]
g: preceded by [n(43), a(13), i(12)]     followed by [e(19), u(10), h(9)]
y: preceded by [l(21), a(13), b(8)]      followed by []
p: preceded by [o(12), u(11), m(10)]     followed by [e(22), r(19), h(18)]
w: preceded by [o(16)]                   followed by [h(20), i(19), a(16)]
b: preceded by [a(14), m(9)]             followed by [e(17), l(14), u(11)]
v: preceded by []                        followed by [e(27), y(5)]
k: preceded by [c(10)]                   followed by [e(14), y(4)]
j: preceded by [b(2)]                    followed by [rare]
x: preceded by [rare]                    followed by [p(3)]
q: preceded by [rare]                    followed by [u(14)]
z: preceded by [z(1)]                    followed by [z(1)]
```

## Most confusable pairs (must be in different groups)

Letters with similar context profiles are hard to distinguish. The top pairs:

```
h-l: similarity=134  (both follow vowels heavily)
l-m: similarity=120  (both follow a, e)
l-d: similarity=118  (both follow a, e)
l-c: similarity=115  (both follow a, e)
d-g: similarity=87   (both follow a, i, n)
d-c: similarity=85   (both follow a, e)
c-m: similarity=82   (both follow a, e)
```

## Optimized groups (68.8% less internal confusion than phone layout)

```
Group 1: [j, m, f, g]  confusion: 91
  j -> rare contexts
  m -> after o, e, a
  f -> after o, l
  g -> after n, a, i

Group 2: [c, q, k, y]  confusion: 27
  c -> after i, a, e
  q -> rare
  k -> after c
  y -> after l, a, b

Group 3: [z, u, l, v]  confusion: 58
  z -> rare
  u -> after o, s, b
  l -> after a, e, i
  v -> rare

Group 4: [b, x, h, w]  confusion: 67
  b -> after a, m
  x -> rare
  h -> after t, c, w
  w -> after o

Group 5: [d, p]  confusion: 35
  d -> after n, e, a
  p -> after o, u, m
```

Total internal confusion (optimized): 278
Total internal confusion (phone keypad): 891
Improvement: 68.8%

## What this means in practice

When the model sees Group 4's token after "t", it knows it's "h" (position 3) because "b", "x", and "w" almost never follow "t". The surrounding context disambiguates within-group letters almost for free.
