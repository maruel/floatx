// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package floatx_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/maruel/floatx"
)

type testData struct {
	V        uint16
	F        float32
	Sign     uint8
	Exponent uint8
	Mantissa uint16
}

func Test_BF16_All(t *testing.T) {
	for i, line := range bf16TestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			if i == 32 {
				t.Skip("bug in denormalized value for bfloat32")
			}
			f := floatx.BF16(line.V)
			testOne8(t, f, line)
			// little endian forever.
			b := [2]byte{byte(line.V), byte(line.V >> 8)}
			if got := floatx.DecodeBF16(b[:]); got != f {
				t.Errorf("%v != %v", got, f)
			}
		})
	}
}

func Test_BF16_SpotCheck(t *testing.T) {
	// Spot check a few values to not take any chance from:
	// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Examples
	data := []struct {
		index int
		want  float64
	}{
		{0x0000, 0.},
		{0x8000, -0.},
		{0x3F80, 1.},
		{0xC000, -2.},
		{0x7F7F, 3.3895313892515355e+38},
		{0x7F80, math.Inf(0)},
		{0x7F81, math.NaN()},
		{0xFF7F, -3.3895313892515355e+38},
		{0xFF80, math.Inf(-1)},
		{0xFF81, math.NaN()},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(bf16TestData[line.index].F); got != line.want && !(math.IsNaN(got) && math.IsNaN(line.want)) {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}

func Test_F16_All(t *testing.T) {
	for i, line := range f16TestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			f := floatx.F16(line.V)
			testOne16(t, f, line)
			// little endian forever.
			b := [2]byte{byte(line.V), byte(line.V >> 8)}
			if got := floatx.DecodeF16(b[:]); got != f {
				t.Errorf("%v != %v", got, f)
			}
		})
	}
}

func Test_F16_SpotCheck(t *testing.T) {
	// Spot check a few values to not take any chance from:
	// https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples
	data := []struct {
		index int
		want  float64
	}{
		{0x0000, 0.},
		{0x8000, -0.},
		{0x3C00, 1.},
		{0xC000, -2.},
		{0x7BFF, 65504},
		{0x7C00, math.Inf(0)},
		{0x7C01, math.NaN()},
		{0xFBFF, -65504},
		{0xFC00, math.Inf(-1)},
		{0xFC01, math.NaN()},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(f16TestData[line.index].F); got != line.want && !(math.IsNaN(got) && math.IsNaN(line.want)) {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}

func Test_F8E4M3_All(t *testing.T) {
	for i, line := range f8E4M3TestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			testOne8(t, floatx.F8E4M3(line.V), line)
		})
	}
}

func Test_F8E4M3_SpotCheck(t *testing.T) {
	// Spot check a few values to not take any chance from:
	// https://en.wikipedia.org/wiki/Minifloat
	data := []struct {
		index int
		want  float64
	}{
		{0x00, 0.},
		{0x80, -0.},
		{0x38, 1.},
		{0xC0, -2.},
		{0x77, 240},
		{0x78, math.Inf(0)},
		{0x79, math.NaN()},
		{0xF7, -240},
		{0xF8, math.Inf(-1)},
		{0xF9, math.NaN()},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(f8E4M3TestData[line.index].F); got != line.want && !(math.IsNaN(got) && math.IsNaN(line.want)) {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}

func Test_F8E4M3Fn_All(t *testing.T) {
	for i, line := range f8E4M3FnTestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			testOne8(t, floatx.F8E4M3Fn(line.V), line)
		})
	}
}

func Test_F8E4M3Fn_SpotCheck(t *testing.T) {
	// Spot check a few values to not take any chance.
	// https://www.tensorflow.org/api_docs/python/tf/dtypes/experimental explains it well.
	data := []struct {
		index int
		want  float64
	}{
		{0x00, 0.},
		{0x38, 1.},
		{0xC0, -2.},
		{0x7E, 448},
		{0x7F, math.NaN()},
		{0xFE, -448},
		{0xFF, math.NaN()},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(f8E4M3FnTestData[line.index].F); got != line.want && !(math.IsNaN(got) && math.IsNaN(line.want)) {
				t.Errorf("b=%x want=%g got=%g", line.index, line.want, got)
			}
		})
	}
}

func Test_F8E5M2_All(t *testing.T) {
	for i, line := range f8E5M2TestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			testOne8(t, floatx.F8E5M2(line.V), line)
		})
	}
}

func Test_F8E5M2_SpotCheck(t *testing.T) {
	// Spot check a few values to not take any chance from:
	// https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
	data := []struct {
		index int
		want  float64
	}{
		{0x00, 0.},
		{0x80, -0.},
		{0x3C, 1.},
		{0xC0, -2.},
		{0x7B, 57344},
		{0x7C, math.Inf(0)},
		{0x7D, math.NaN()},
		{0xFB, -57344},
		{0xFC, math.Inf(-1)},
		{0xFD, math.NaN()},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(f8E5M2TestData[line.index].F); got != line.want && !(math.IsNaN(got) && math.IsNaN(line.want)) {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}

type fn8 interface {
	Components() (uint8, uint8, uint8)
	Float32() float32
}

type fn16 interface {
	Components() (uint8, uint8, uint16)
	Float32() float32
}

func testOne8[T fn8](t *testing.T, f T, line testData) {
	sign, exponent, mantissa := f.Components()
	if sign != line.Sign {
		t.Errorf("sign: want=%d  got=%d", line.Sign, sign)
	}
	if exponent != line.Exponent {
		t.Errorf("exponent: want=%d  got=%d", line.Exponent, exponent)
	}
	if uint16(mantissa) != line.Mantissa {
		t.Errorf("mantissa: want=%d  got=%d", line.Mantissa, mantissa)
	}
	if got := f.Float32(); got != line.F {
		if !math.IsNaN(float64(got)) && !math.IsNaN(float64(line.F)) {
			t.Errorf("%g != %g", got, line.F)
		}
	}
}

func testOne16[T fn16](t *testing.T, f T, line testData) {
	sign, exponent, mantissa := f.Components()
	if sign != line.Sign {
		t.Errorf("sign: want=%d  got=%d", line.Sign, sign)
	}
	if exponent != line.Exponent {
		t.Errorf("exponent: want=%d  got=%d", line.Exponent, exponent)
	}
	if mantissa != line.Mantissa {
		t.Errorf("mantissa: want=%d  got=%d", line.Mantissa, mantissa)
	}
	if got := f.Float32(); got != line.F {
		if !math.IsNaN(float64(got)) && !math.IsNaN(float64(line.F)) {
			t.Errorf("%g != %g", got, line.F)
		}
	}
}

// Not too large so it doesn't trash the cache.
var largeArray = make([]byte, 1024)

var benchmarkResultBF16 floatx.BF16

func Benchmark_DecodeBF16(b *testing.B) {
	var dummy floatx.BF16
	for i := range b.N {
		offset := (2 * i) % (len(largeArray) - 2)
		dummy += floatx.DecodeBF16(largeArray[offset:])
	}
	benchmarkResultBF16 = dummy
}

var benchmarkResultF16 floatx.F16

func Benchmark_DecodeF16(b *testing.B) {
	var dummy floatx.F16
	for i := range b.N {
		offset := (2 * i) % (len(largeArray) - 2)
		dummy += floatx.DecodeF16(largeArray[offset:])
	}
	benchmarkResultF16 = dummy
}

var benchmarkResultFloat float32

func Benchmark_BF16_Float32(b *testing.B) {
	var dummy float32
	for i := range b.N {
		dummy += floatx.BF16(uint16(i)).Float32()
	}
	benchmarkResultFloat = dummy
}

func Benchmark_F16_Float32(b *testing.B) {
	var dummy float32
	for i := range b.N {
		dummy += floatx.F16(uint16(i)).Float32()
	}
	benchmarkResultFloat = dummy
}

func Benchmark_F8E4M3_Float32(b *testing.B) {
	var dummy float32
	for i := range b.N {
		dummy += floatx.F8E4M3(uint8(i)).Float32()
	}
	benchmarkResultFloat = dummy
}

func Benchmark_F8E4M3Fn_Float32(b *testing.B) {
	var dummy float32
	for i := range b.N {
		dummy += floatx.F8E4M3Fn(uint8(i)).Float32()
	}
	benchmarkResultFloat = dummy
}

func Benchmark_F8E55M2_Float32(b *testing.B) {
	var dummy float32
	for i := range b.N {
		dummy += floatx.F8E5M2(uint8(i)).Float32()
	}
	benchmarkResultFloat = dummy
}
