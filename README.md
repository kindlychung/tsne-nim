# TSNE implementation with Nim

This is port from python implementation [here](https://lvdmaaten.github.io/tsne/).

This should work for any arraymancer tensor if you pass arguments correctly.
Please see main of tsne.nim.

This is for my test implementation. Though I've confirmed the result is almost same as python's one, I don't guarantee rogical correctness.

## Requirements

- Nim 1.0 or higher
- Arraymancer 0.20

## Build

```
$ nimble build
```

If you need high performance,

```
$ nimble build -d:danger -d:release
```

then

```
$ ./result/bin/tsne
```
