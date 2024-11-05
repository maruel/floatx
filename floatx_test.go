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
			bf := floatx.BF16(line.V)
			sign, exponent, mantissa := bf.Components()
			if sign != line.Sign {
				t.Errorf("sign: want=%d  got=%d", line.Sign, sign)
			}
			if exponent != line.Exponent {
				t.Errorf("exponent: want=%d  got=%d", line.Exponent, exponent)
			}
			if uint16(mantissa) != line.Mantissa {
				t.Errorf("mantissa: want=%d  got=%d", line.Mantissa, mantissa)
			}
			if got := bf.Float32(); got != line.F {
				if !math.IsNaN(float64(got)) && !math.IsNaN(float64(line.F)) {
					t.Errorf("%g != %g", got, line.F)
				}
			}
			// little endian forever.
			b := [2]byte{byte(line.V), byte(line.V >> 8)}
			if got := floatx.DecodeBF16(b[:]); got != bf {
				t.Errorf("%v != %v", got, bf)
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
		{0x7F80, math.Inf(0)},
		{0xFF80, math.Inf(-1)},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(bf16TestData[line.index].F); got != line.want {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}

func Test_F16_All(t *testing.T) {
	for i, line := range f16TestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			f := floatx.F16(line.V)
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
		{0x7C00, math.Inf(0)},
		{0xFC00, math.Inf(-1)},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(f16TestData[line.index].F); got != line.want {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}

func Test_F8E4M3_All(t *testing.T) {
	for i, line := range f8E4M3TestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			f := floatx.F8E4M3(line.V)
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
		{0x78, math.Inf(0)},
		{0xF8, math.Inf(-1)},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(f8E4M3TestData[line.index].F); got != line.want {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}

func Test_F8E5M2_All(t *testing.T) {
	for i, line := range f8E5M2TestData {
		t.Run(fmt.Sprintf("#%d: %g", i, line.F), func(t *testing.T) {
			f := floatx.F8E5M2(line.V)
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
		})
	}
}

func Test_F8E5M2_SpotCheck(t *testing.T) {
	// Spot check a few values to not take any chance from:
	// https://en.wikipedia.org/wiki/Minifloat
	data := []struct {
		index int
		want  float64
	}{
		{0x00, 0.},
		{0x80, -0.},
		{0x3C, 1.},
		{0xC0, -2.},
		{0x7C, math.Inf(0)},
		{0xFC, math.Inf(-1)},
	}
	for i, line := range data {
		t.Run(fmt.Sprintf("#%d: %g", i, line.want), func(t *testing.T) {
			if got := float64(f8E5M2TestData[line.index].F); got != line.want {
				t.Errorf("want=%g got=%g", line.want, got)
			}
		})
	}
}
