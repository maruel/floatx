// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

//go:generate go run gen.go

// Package floatx implements various floating encodings with 100% code coverage.
package floatx

import (
	"encoding/binary"
	"math"
)

// F32

// F32 bit allocation.
const (
	// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
	F32SignOffset     = 31
	F32ExponentOffset = 23
	F32ExponentBias   = (1<<(F32SignOffset-F32ExponentOffset))/2 - 1
	F32ExponentMask   = (1 << (F32SignOffset - F32ExponentOffset)) - 1
	F32MantissaMask   = (1 << F32ExponentOffset) - 1
)

// F32 is a float32.
//
// The only use case is to call Components() on it.
//
// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
type F32 float32

// Components returns the sign, exponent and mantissa bits separated.
func (f F32) Components() (uint8, uint8, uint32) {
	b := math.Float32bits(float32(f))
	sign := b >> F32SignOffset
	exponent := (b >> F32ExponentOffset) & F32ExponentMask
	mantissa := b & F32MantissaMask
	return uint8(sign), uint8(exponent), uint32(mantissa)
}

// BF16

// BF16 bit allocation.
const (
	BF16SignOffset     = 15
	BF16ExponentOffset = 7
	BF16ExponentMask   = (1 << (BF16SignOffset - BF16ExponentOffset)) - 1
	BF16ExponentBias   = (1<<(BF16SignOffset-BF16ExponentOffset))/2 - 1
	BF16MantissaMask   = (1 << BF16ExponentOffset) - 1
)

// BF16 represents a Google Brain float 16, or bfloat16.
//
// It is equivalent to torch.bfloat16 or
// https://github.com/jax-ml/ml_dtypes#bfloat16.
//
// See https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
type BF16 uint16

// DecodeBF16 decode a little endian value.
func DecodeBF16(b []byte) BF16 {
	return BF16(binary.LittleEndian.Uint16(b))
}

// Components returns the sign, exponent and mantissa bits separated.
func (b BF16) Components() (uint8, uint8, uint8) {
	sign := b >> BF16SignOffset
	exponent := (b >> BF16ExponentOffset) & BF16ExponentMask
	mantissa := b & BF16MantissaMask
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (b BF16) Float32() float32 {
	sign8, exponent8, mantissa8 := b.Components()
	// Realign sign right away.
	sign := uint32(sign8) << F32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 7 bits in bfloat16 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (F32ExponentOffset - BF16ExponentOffset)
	if exponent == BF16ExponentMask {
		// Either Inf or NaN.
		// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Positive_and_negative_infinity
		// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Not_a_Number
		return math.Float32frombits(sign | (F32ExponentMask << F32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding
		exponent++
		for mantissa&(BF16ExponentMask<<F32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= F32MantissaMask
	}
	exponent += F32ExponentBias - BF16ExponentBias
	return math.Float32frombits(sign | (exponent << F32ExponentOffset) | mantissa)
}

// F16

// F16 bit allocation.
const (
	F16SignOffset     = 15
	F16ExponentOffset = 10
	F16ExponentMask   = (1 << (F16SignOffset - F16ExponentOffset)) - 1
	F16ExponentBias   = (1<<(F16SignOffset-F16ExponentOffset))/2 - 1
	F16MantissaMask   = (1 << F16ExponentOffset) - 1
)

// F16 represents a IEEE 754 half-precision binary floating-point format
//
// It is equivalent to torch.float16.
//
// See https://en.wikipedia.org/wiki/Half-precision_floating-point_format
type F16 uint16

// DecodeF16 decode a little endian value.
func DecodeF16(b []byte) F16 {
	return F16(binary.LittleEndian.Uint16(b))
}

// Components returns the sign, exponent and mantissa bits separated.
func (f F16) Components() (uint8, uint8, uint16) {
	sign := f >> F16SignOffset
	exponent := (f >> F16ExponentOffset) & F16ExponentMask
	mantissa := f & F16MantissaMask
	return uint8(sign), uint8(exponent), uint16(mantissa)
}

// Float32 returns the float32 equivalent.
func (f F16) Float32() float32 {
	sign8, exponent8, mantissa8 := f.Components()
	// Realign sign right away.
	sign := uint32(sign8) << F32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 10 bits in float16 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (F32ExponentOffset - F16ExponentOffset)
	if exponent == F16ExponentMask {
		// Either Inf or NaN.
		return math.Float32frombits(sign | (F32ExponentMask << F32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		// https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
		exponent++
		for mantissa&(F16ExponentMask<<F32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= F32MantissaMask
	}
	exponent += F32ExponentBias - F16ExponentBias
	return math.Float32frombits(sign | (exponent << F32ExponentOffset) | mantissa)
}

// F8E4M3

// F8E4M3 and F8E4M3Fn bit allocation.
const (
	F8E4M3SignOffset     = 7
	F8E4M3ExponentOffset = 3
	F8E4M3ExponentMask   = (1 << (F8E4M3SignOffset - F8E4M3ExponentOffset)) - 1
	F8E4M3ExponentBias   = (1<<(F8E4M3SignOffset-F8E4M3ExponentOffset))/2 - 1
	F8E4M3MantissaMask   = (1 << F8E4M3ExponentOffset) - 1
)

// F8E4M3 represents a float8 with 4 exponent bits and 3 mantissa bits.
//
// It is consistent with IEEE 754.
//
// It can store values up to +/-240, +/- inf and nan.
//
// See https://en.wikipedia.org/wiki/Minifloat
type F8E4M3 uint8

// Components returns the sign, exponent and mantissa bits separated.
func (f F8E4M3) Components() (uint8, uint8, uint8) {
	sign := f >> F8E4M3SignOffset
	exponent := (f >> F8E4M3ExponentOffset) & F8E4M3ExponentMask
	mantissa := f & F8E4M3MantissaMask
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (f F8E4M3) Float32() float32 {
	sign8, exponent8, mantissa8 := f.Components()
	// Realign sign right away.
	sign := uint32(sign8) << F32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 3 bits in float8 E4M3 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (F32ExponentOffset - F8E4M3ExponentOffset)
	if exponent == F8E4M3ExponentMask {
		// Either Inf or NaN.
		return math.Float32frombits(sign | (F32ExponentMask << F32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		exponent++
		for mantissa&(F8E4M3ExponentMask<<F32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= F32MantissaMask
	}
	exponent += F32ExponentBias - F8E4M3ExponentBias
	return math.Float32frombits(sign | (exponent << F32ExponentOffset) | mantissa)
}

// F8E4M3Fn

// F8E4M3Fn represents a float8 with 4 exponent bits and 3 mantissa bits.
//
// It can store values up to +/-448 and nan. It cannot store inf.
//
// See https://github.com/jax-ml/ml_dtypes#float8_e4m3fn
type F8E4M3Fn uint8

// Components returns the sign, exponent and mantissa bits separated.
func (f F8E4M3Fn) Components() (uint8, uint8, uint8) {
	sign := f >> F8E4M3SignOffset
	exponent := (f >> F8E4M3ExponentOffset) & F8E4M3ExponentMask
	mantissa := f & F8E4M3MantissaMask
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (f F8E4M3Fn) Float32() float32 {
	sign8, exponent8, mantissa8 := f.Components()
	// Realign sign right away.
	sign := uint32(sign8) << F32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 3 bits in float8 E4M3 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (F32ExponentOffset - F8E4M3ExponentOffset)
	if f == 0x7F || f == 0xFF {
		// Positive and negative NaN
		return float32(math.NaN())
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		exponent++
		for mantissa&(F8E4M3ExponentMask<<F32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= F32MantissaMask
	}
	exponent += F32ExponentBias - F8E4M3ExponentBias
	return math.Float32frombits(sign | (exponent << F32ExponentOffset) | mantissa)
}

// F8E5M2

// F8E5M2 bit allocation.
const (
	F8E5M2SignOffset     = 7
	F8E5M2ExponentOffset = 2
	F8E5M2ExponentMask   = (1 << (F8E5M2SignOffset - F8E5M2ExponentOffset)) - 1
	F8E5M2ExponentBias   = (1<<(F8E5M2SignOffset-F8E5M2ExponentOffset))/2 - 1
	F8E5M2MantissaMask   = (1 << F8E5M2ExponentOffset) - 1
)

// F8E5M2 represents a float8 with 5 exponent bits and 2 mantissa bits.
//
// It can store values up to +/-57344, +/- inf and nan.
//
// See https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
type F8E5M2 uint8

// Components returns the sign, exponent and mantissa bits separated.
func (f F8E5M2) Components() (uint8, uint8, uint8) {
	sign := f >> F8E5M2SignOffset
	exponent := (f >> F8E5M2ExponentOffset) & F8E5M2ExponentMask
	mantissa := f & F8E5M2MantissaMask
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (f F8E5M2) Float32() float32 {
	sign8, exponent8, mantissa8 := f.Components()
	// Realign sign right away.
	sign := uint32(sign8) << F32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 2 bits in float8 E5M2 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (F32ExponentOffset - F8E5M2ExponentOffset)
	if exponent == F8E5M2ExponentMask {
		// Either Inf or NaN.
		return math.Float32frombits(sign | (F32ExponentMask << F32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		exponent++
		for mantissa&(F8E5M2ExponentMask<<F32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= F32MantissaMask
	}
	exponent += F32ExponentBias - F8E5M2ExponentBias
	return math.Float32frombits(sign | (exponent << F32ExponentOffset) | mantissa)
}
