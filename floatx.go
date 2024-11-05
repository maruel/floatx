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

// BF16

const (
	bf16SignOffset     = 15
	bf16ExponentOffset = 7
)

// BF16 represents a google brain 16 float.
//
// See https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
type BF16 uint16

// DecodeBF16 decode a little endian value.
func DecodeBF16(b []byte) BF16 {
	return BF16(binary.LittleEndian.Uint16(b))
}

// Components returns the sign, exponent and mantissa bits separated.
func (b BF16) Components() (uint8, uint8, uint8) {
	const exponentMask = (1 << (bf16SignOffset - bf16ExponentOffset)) - 1
	const mantissaMask = (1 << bf16ExponentOffset) - 1
	sign := b >> bf16SignOffset
	exponent := (b >> bf16ExponentOffset) & exponentMask
	mantissa := b & mantissaMask
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (b BF16) Float32() float32 {
	const exponentMask = (1 << (bf16SignOffset - bf16ExponentOffset)) - 1
	const exponentBias = (1<<(bf16SignOffset-bf16ExponentOffset))/2 - 1
	const f32exponentMask = (1 << (f32SignOffset - f32ExponentOffset)) - 1
	sign8, exponent8, mantissa8 := b.Components()
	// Realign sign right away.
	sign := uint32(sign8) << f32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 7 bits in bfloat16 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (f32ExponentOffset - bf16ExponentOffset)
	if exponent == exponentMask {
		// Either Inf or NaN.
		// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Positive_and_negative_infinity
		// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Not_a_Number
		return math.Float32frombits(sign | (f32exponentMask << f32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		// https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding
		exponent++
		for mantissa&(exponentMask<<f32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= (1 << f32ExponentOffset) - 1
	}
	exponent += f32ExponentBias - exponentBias
	return math.Float32frombits(sign | (exponent << f32ExponentOffset) | mantissa)
}

// F16

const (
	f16SignOffset     = 15
	f16ExponentOffset = 10
)

// F16 represents a IEEE 754 half-precision binary floating-point format
//
// See https://en.wikipedia.org/wiki/Half-precision_floating-point_format
type F16 uint16

// DecodeF16 decode a little endian value.
func DecodeF16(b []byte) F16 {
	return F16(binary.LittleEndian.Uint16(b))
}

// Components returns the sign, exponent and mantissa bits separated.
func (f F16) Components() (uint8, uint8, uint16) {
	const exponentMask = (1 << (f16SignOffset - f16ExponentOffset)) - 1
	const mantissaMask = (1 << f16ExponentOffset) - 1
	sign := f >> f16SignOffset
	exponent := (f >> f16ExponentOffset) & exponentMask
	mantissa := f & mantissaMask
	return uint8(sign), uint8(exponent), uint16(mantissa)
}

// Float32 returns the float32 equivalent.
func (f F16) Float32() float32 {
	const exponentMask = (1 << (f16SignOffset - f16ExponentOffset)) - 1
	const exponentBias = (1<<(f16SignOffset-f16ExponentOffset))/2 - 1
	const f32exponentMask = (1 << (f32SignOffset - f32ExponentOffset)) - 1
	sign8, exponent8, mantissa8 := f.Components()
	// Realign sign right away.
	sign := uint32(sign8) << f32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 10 bits in float16 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (f32ExponentOffset - f16ExponentOffset)
	if exponent == exponentMask {
		// Either Inf or NaN.
		return math.Float32frombits(sign | (f32exponentMask << f32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		// https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Exponent_encoding
		exponent++
		for mantissa&(exponentMask<<f32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= (1 << f32ExponentOffset) - 1
	}
	exponent += f32ExponentBias - exponentBias
	return math.Float32frombits(sign | (exponent << f32ExponentOffset) | mantissa)
}

// F32

const (
	// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
	f32SignOffset     = 31
	f32ExponentOffset = 23
	f32ExponentBias   = 127
)

// F8E4M3

const (
	f8E4M3SignOffset     = 7
	f8E4M3ExponentOffset = 3
)

// F8E4M3 represents a reduced float8 with 4 bits of exponent and 3 bits of
// mantissa.
//
// See https://en.wikipedia.org/wiki/Minifloat
type F8E4M3 uint8

// Components returns the sign, exponent and mantissa bits separated.
func (f F8E4M3) Components() (uint8, uint8, uint8) {
	const exponentMask = (1 << (f8E4M3SignOffset - f8E4M3ExponentOffset)) - 1
	const mantissaMask = (1 << f8E4M3ExponentOffset) - 1
	sign := f >> f8E4M3SignOffset
	exponent := (f >> f8E4M3ExponentOffset) & exponentMask
	mantissa := f & mantissaMask
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (f F8E4M3) Float32() float32 {
	const exponentMask = (1 << (f8E4M3SignOffset - f8E4M3ExponentOffset)) - 1
	const exponentBias = (1<<(f8E4M3SignOffset-f8E4M3ExponentOffset))/2 - 1
	const f32exponentMask = (1 << (f32SignOffset - f32ExponentOffset)) - 1
	sign8, exponent8, mantissa8 := f.Components()
	// Realign sign right away.
	sign := uint32(sign8) << f32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 3 bits in float8 E4M3 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (f32ExponentOffset - f8E4M3ExponentOffset)
	if exponent == exponentMask {
		// Either Inf or NaN.
		return math.Float32frombits(sign | (f32exponentMask << f32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		exponent++
		for mantissa&(exponentMask<<f32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= (1 << f32ExponentOffset) - 1
	}
	exponent += f32ExponentBias - exponentBias
	return math.Float32frombits(sign | (exponent << f32ExponentOffset) | mantissa)
}

// F8E5M2

const (
	f8E5M2SignOffset     = 7
	f8E5M2ExponentOffset = 2
)

// F8E5M2 represents a reduced float8 with 5 bits of exponent and 2 bits of
// mantissa.
//
// See https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
type F8E5M2 uint8

// Components returns the sign, exponent and mantissa bits separated.
func (f F8E5M2) Components() (uint8, uint8, uint8) {
	const exponentMask = (1 << (f8E5M2SignOffset - f8E5M2ExponentOffset)) - 1
	const mantissaMask = (1 << f8E5M2ExponentOffset) - 1
	sign := f >> f8E5M2SignOffset
	exponent := (f >> f8E5M2ExponentOffset) & exponentMask
	mantissa := f & mantissaMask
	return uint8(sign), uint8(exponent), uint8(mantissa)
}

// Float32 returns the float32 equivalent.
func (f F8E5M2) Float32() float32 {
	const exponentMask = (1 << (f8E5M2SignOffset - f8E5M2ExponentOffset)) - 1
	const exponentBias = (1<<(f8E5M2SignOffset-f8E5M2ExponentOffset))/2 - 1
	const f32exponentMask = (1 << (f32SignOffset - f32ExponentOffset)) - 1
	sign8, exponent8, mantissa8 := f.Components()
	// Realign sign right away.
	sign := uint32(sign8) << f32SignOffset
	exponent := uint32(exponent8)
	// Realign mantissa right away. The fraction is 2 bits in float8 E5M2 and 23 bits in float32.
	mantissa := uint32(mantissa8) << (f32ExponentOffset - f8E5M2ExponentOffset)
	if exponent == exponentMask {
		// Either Inf or NaN.
		return math.Float32frombits(sign | (f32exponentMask << f32ExponentOffset) | mantissa)
	}
	// If no exponent.
	if exponent == 0 {
		if mantissa == 0 {
			return math.Float32frombits(sign)
		}
		// Normalize subnormal numbers.
		exponent++
		for mantissa&(exponentMask<<f32ExponentOffset) == 0 {
			mantissa <<= 1
			exponent--
		}
		mantissa &= (1 << f32ExponentOffset) - 1
	}
	exponent += f32ExponentBias - exponentBias
	return math.Float32frombits(sign | (exponent << f32ExponentOffset) | mantissa)
}
