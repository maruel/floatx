# floatx

Implementation of floating point number types often used in deep neural networks (DNN).

- float8 E4M3 [FP8E4M3](https://pkg.go.dev/github.com/maruel/floatx#FP8E4M3)
- float8 E4M3Fn [FP8E4M3Fn](https://pkg.go.dev/github.com/maruel/floatx#FP8E4M3Fn)
- float8 E5M2 [FP8E5M2](https://pkg.go.dev/github.com/maruel/floatx#FP8E5M2)
- float16 [F16](https://pkg.go.dev/github.com/maruel/floatx#F16)
- bfloat16 [BF16](https://pkg.go.dev/github.com/maruel/floatx#BF16)

See whole documentation at [![Go Reference](https://pkg.go.dev/badge/github.com/maruel/floatx/.svg)](https://pkg.go.dev/github.com/maruel/floatx/)

No external dependency.

Has 100% code coverage:
[![codecov](https://codecov.io/gh/maruel/floatx/graph/badge.svg?token=M92Q70R7BZ)](https://codecov.io/gh/maruel/floatx)
including testing the whole 256 possibilities for float8 and 65536 values for float16 types.
