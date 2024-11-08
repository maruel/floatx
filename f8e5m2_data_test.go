// Code generated "go run gen.go" DO NOT EDIT.

package floatx_test

import "math"

// See floatx_test.go
var f8E5M2TestData = []testData{
	{0x00, 0, 0, 0, 0},
	{0x01, 1.5258789e-05, 0, 0, 1},
	{0x02, 3.0517578e-05, 0, 0, 2},
	{0x03, 4.5776367e-05, 0, 0, 3},
	{0x04, 6.1035156e-05, 0, 1, 0},
	{0x05, 7.6293945e-05, 0, 1, 1},
	{0x06, 9.1552734e-05, 0, 1, 2},
	{0x07, 0.00010681152, 0, 1, 3},
	{0x08, 0.00012207031, 0, 2, 0},
	{0x09, 0.00015258789, 0, 2, 1},
	{0x0a, 0.00018310547, 0, 2, 2},
	{0x0b, 0.00021362305, 0, 2, 3},
	{0x0c, 0.00024414062, 0, 3, 0},
	{0x0d, 0.00030517578, 0, 3, 1},
	{0x0e, 0.00036621094, 0, 3, 2},
	{0x0f, 0.0004272461, 0, 3, 3},
	{0x10, 0.00048828125, 0, 4, 0},
	{0x11, 0.00061035156, 0, 4, 1},
	{0x12, 0.0007324219, 0, 4, 2},
	{0x13, 0.0008544922, 0, 4, 3},
	{0x14, 0.0009765625, 0, 5, 0},
	{0x15, 0.0012207031, 0, 5, 1},
	{0x16, 0.0014648438, 0, 5, 2},
	{0x17, 0.0017089844, 0, 5, 3},
	{0x18, 0.001953125, 0, 6, 0},
	{0x19, 0.0024414062, 0, 6, 1},
	{0x1a, 0.0029296875, 0, 6, 2},
	{0x1b, 0.0034179688, 0, 6, 3},
	{0x1c, 0.00390625, 0, 7, 0},
	{0x1d, 0.0048828125, 0, 7, 1},
	{0x1e, 0.005859375, 0, 7, 2},
	{0x1f, 0.0068359375, 0, 7, 3},
	{0x20, 0.0078125, 0, 8, 0},
	{0x21, 0.009765625, 0, 8, 1},
	{0x22, 0.01171875, 0, 8, 2},
	{0x23, 0.013671875, 0, 8, 3},
	{0x24, 0.015625, 0, 9, 0},
	{0x25, 0.01953125, 0, 9, 1},
	{0x26, 0.0234375, 0, 9, 2},
	{0x27, 0.02734375, 0, 9, 3},
	{0x28, 0.03125, 0, 10, 0},
	{0x29, 0.0390625, 0, 10, 1},
	{0x2a, 0.046875, 0, 10, 2},
	{0x2b, 0.0546875, 0, 10, 3},
	{0x2c, 0.0625, 0, 11, 0},
	{0x2d, 0.078125, 0, 11, 1},
	{0x2e, 0.09375, 0, 11, 2},
	{0x2f, 0.109375, 0, 11, 3},
	{0x30, 0.125, 0, 12, 0},
	{0x31, 0.15625, 0, 12, 1},
	{0x32, 0.1875, 0, 12, 2},
	{0x33, 0.21875, 0, 12, 3},
	{0x34, 0.25, 0, 13, 0},
	{0x35, 0.3125, 0, 13, 1},
	{0x36, 0.375, 0, 13, 2},
	{0x37, 0.4375, 0, 13, 3},
	{0x38, 0.5, 0, 14, 0},
	{0x39, 0.625, 0, 14, 1},
	{0x3a, 0.75, 0, 14, 2},
	{0x3b, 0.875, 0, 14, 3},
	{0x3c, 1, 0, 15, 0},
	{0x3d, 1.25, 0, 15, 1},
	{0x3e, 1.5, 0, 15, 2},
	{0x3f, 1.75, 0, 15, 3},
	{0x40, 2, 0, 16, 0},
	{0x41, 2.5, 0, 16, 1},
	{0x42, 3, 0, 16, 2},
	{0x43, 3.5, 0, 16, 3},
	{0x44, 4, 0, 17, 0},
	{0x45, 5, 0, 17, 1},
	{0x46, 6, 0, 17, 2},
	{0x47, 7, 0, 17, 3},
	{0x48, 8, 0, 18, 0},
	{0x49, 10, 0, 18, 1},
	{0x4a, 12, 0, 18, 2},
	{0x4b, 14, 0, 18, 3},
	{0x4c, 16, 0, 19, 0},
	{0x4d, 20, 0, 19, 1},
	{0x4e, 24, 0, 19, 2},
	{0x4f, 28, 0, 19, 3},
	{0x50, 32, 0, 20, 0},
	{0x51, 40, 0, 20, 1},
	{0x52, 48, 0, 20, 2},
	{0x53, 56, 0, 20, 3},
	{0x54, 64, 0, 21, 0},
	{0x55, 80, 0, 21, 1},
	{0x56, 96, 0, 21, 2},
	{0x57, 112, 0, 21, 3},
	{0x58, 128, 0, 22, 0},
	{0x59, 160, 0, 22, 1},
	{0x5a, 192, 0, 22, 2},
	{0x5b, 224, 0, 22, 3},
	{0x5c, 256, 0, 23, 0},
	{0x5d, 320, 0, 23, 1},
	{0x5e, 384, 0, 23, 2},
	{0x5f, 448, 0, 23, 3},
	{0x60, 512, 0, 24, 0},
	{0x61, 640, 0, 24, 1},
	{0x62, 768, 0, 24, 2},
	{0x63, 896, 0, 24, 3},
	{0x64, 1024, 0, 25, 0},
	{0x65, 1280, 0, 25, 1},
	{0x66, 1536, 0, 25, 2},
	{0x67, 1792, 0, 25, 3},
	{0x68, 2048, 0, 26, 0},
	{0x69, 2560, 0, 26, 1},
	{0x6a, 3072, 0, 26, 2},
	{0x6b, 3584, 0, 26, 3},
	{0x6c, 4096, 0, 27, 0},
	{0x6d, 5120, 0, 27, 1},
	{0x6e, 6144, 0, 27, 2},
	{0x6f, 7168, 0, 27, 3},
	{0x70, 8192, 0, 28, 0},
	{0x71, 10240, 0, 28, 1},
	{0x72, 12288, 0, 28, 2},
	{0x73, 14336, 0, 28, 3},
	{0x74, 16384, 0, 29, 0},
	{0x75, 20480, 0, 29, 1},
	{0x76, 24576, 0, 29, 2},
	{0x77, 28672, 0, 29, 3},
	{0x78, 32768, 0, 30, 0},
	{0x79, 40960, 0, 30, 1},
	{0x7a, 49152, 0, 30, 2},
	{0x7b, 57344, 0, 30, 3},
	{0x7c, float32(math.Inf(0)), 0, 31, 0},
	{0x7d, float32(math.NaN()), 0, 31, 1},
	{0x7e, float32(math.NaN()), 0, 31, 2},
	{0x7f, float32(math.NaN()), 0, 31, 3},
	{0x80, -0, 1, 0, 0},
	{0x81, -1.5258789e-05, 1, 0, 1},
	{0x82, -3.0517578e-05, 1, 0, 2},
	{0x83, -4.5776367e-05, 1, 0, 3},
	{0x84, -6.1035156e-05, 1, 1, 0},
	{0x85, -7.6293945e-05, 1, 1, 1},
	{0x86, -9.1552734e-05, 1, 1, 2},
	{0x87, -0.00010681152, 1, 1, 3},
	{0x88, -0.00012207031, 1, 2, 0},
	{0x89, -0.00015258789, 1, 2, 1},
	{0x8a, -0.00018310547, 1, 2, 2},
	{0x8b, -0.00021362305, 1, 2, 3},
	{0x8c, -0.00024414062, 1, 3, 0},
	{0x8d, -0.00030517578, 1, 3, 1},
	{0x8e, -0.00036621094, 1, 3, 2},
	{0x8f, -0.0004272461, 1, 3, 3},
	{0x90, -0.00048828125, 1, 4, 0},
	{0x91, -0.00061035156, 1, 4, 1},
	{0x92, -0.0007324219, 1, 4, 2},
	{0x93, -0.0008544922, 1, 4, 3},
	{0x94, -0.0009765625, 1, 5, 0},
	{0x95, -0.0012207031, 1, 5, 1},
	{0x96, -0.0014648438, 1, 5, 2},
	{0x97, -0.0017089844, 1, 5, 3},
	{0x98, -0.001953125, 1, 6, 0},
	{0x99, -0.0024414062, 1, 6, 1},
	{0x9a, -0.0029296875, 1, 6, 2},
	{0x9b, -0.0034179688, 1, 6, 3},
	{0x9c, -0.00390625, 1, 7, 0},
	{0x9d, -0.0048828125, 1, 7, 1},
	{0x9e, -0.005859375, 1, 7, 2},
	{0x9f, -0.0068359375, 1, 7, 3},
	{0xa0, -0.0078125, 1, 8, 0},
	{0xa1, -0.009765625, 1, 8, 1},
	{0xa2, -0.01171875, 1, 8, 2},
	{0xa3, -0.013671875, 1, 8, 3},
	{0xa4, -0.015625, 1, 9, 0},
	{0xa5, -0.01953125, 1, 9, 1},
	{0xa6, -0.0234375, 1, 9, 2},
	{0xa7, -0.02734375, 1, 9, 3},
	{0xa8, -0.03125, 1, 10, 0},
	{0xa9, -0.0390625, 1, 10, 1},
	{0xaa, -0.046875, 1, 10, 2},
	{0xab, -0.0546875, 1, 10, 3},
	{0xac, -0.0625, 1, 11, 0},
	{0xad, -0.078125, 1, 11, 1},
	{0xae, -0.09375, 1, 11, 2},
	{0xaf, -0.109375, 1, 11, 3},
	{0xb0, -0.125, 1, 12, 0},
	{0xb1, -0.15625, 1, 12, 1},
	{0xb2, -0.1875, 1, 12, 2},
	{0xb3, -0.21875, 1, 12, 3},
	{0xb4, -0.25, 1, 13, 0},
	{0xb5, -0.3125, 1, 13, 1},
	{0xb6, -0.375, 1, 13, 2},
	{0xb7, -0.4375, 1, 13, 3},
	{0xb8, -0.5, 1, 14, 0},
	{0xb9, -0.625, 1, 14, 1},
	{0xba, -0.75, 1, 14, 2},
	{0xbb, -0.875, 1, 14, 3},
	{0xbc, -1, 1, 15, 0},
	{0xbd, -1.25, 1, 15, 1},
	{0xbe, -1.5, 1, 15, 2},
	{0xbf, -1.75, 1, 15, 3},
	{0xc0, -2, 1, 16, 0},
	{0xc1, -2.5, 1, 16, 1},
	{0xc2, -3, 1, 16, 2},
	{0xc3, -3.5, 1, 16, 3},
	{0xc4, -4, 1, 17, 0},
	{0xc5, -5, 1, 17, 1},
	{0xc6, -6, 1, 17, 2},
	{0xc7, -7, 1, 17, 3},
	{0xc8, -8, 1, 18, 0},
	{0xc9, -10, 1, 18, 1},
	{0xca, -12, 1, 18, 2},
	{0xcb, -14, 1, 18, 3},
	{0xcc, -16, 1, 19, 0},
	{0xcd, -20, 1, 19, 1},
	{0xce, -24, 1, 19, 2},
	{0xcf, -28, 1, 19, 3},
	{0xd0, -32, 1, 20, 0},
	{0xd1, -40, 1, 20, 1},
	{0xd2, -48, 1, 20, 2},
	{0xd3, -56, 1, 20, 3},
	{0xd4, -64, 1, 21, 0},
	{0xd5, -80, 1, 21, 1},
	{0xd6, -96, 1, 21, 2},
	{0xd7, -112, 1, 21, 3},
	{0xd8, -128, 1, 22, 0},
	{0xd9, -160, 1, 22, 1},
	{0xda, -192, 1, 22, 2},
	{0xdb, -224, 1, 22, 3},
	{0xdc, -256, 1, 23, 0},
	{0xdd, -320, 1, 23, 1},
	{0xde, -384, 1, 23, 2},
	{0xdf, -448, 1, 23, 3},
	{0xe0, -512, 1, 24, 0},
	{0xe1, -640, 1, 24, 1},
	{0xe2, -768, 1, 24, 2},
	{0xe3, -896, 1, 24, 3},
	{0xe4, -1024, 1, 25, 0},
	{0xe5, -1280, 1, 25, 1},
	{0xe6, -1536, 1, 25, 2},
	{0xe7, -1792, 1, 25, 3},
	{0xe8, -2048, 1, 26, 0},
	{0xe9, -2560, 1, 26, 1},
	{0xea, -3072, 1, 26, 2},
	{0xeb, -3584, 1, 26, 3},
	{0xec, -4096, 1, 27, 0},
	{0xed, -5120, 1, 27, 1},
	{0xee, -6144, 1, 27, 2},
	{0xef, -7168, 1, 27, 3},
	{0xf0, -8192, 1, 28, 0},
	{0xf1, -10240, 1, 28, 1},
	{0xf2, -12288, 1, 28, 2},
	{0xf3, -14336, 1, 28, 3},
	{0xf4, -16384, 1, 29, 0},
	{0xf5, -20480, 1, 29, 1},
	{0xf6, -24576, 1, 29, 2},
	{0xf7, -28672, 1, 29, 3},
	{0xf8, -32768, 1, 30, 0},
	{0xf9, -40960, 1, 30, 1},
	{0xfa, -49152, 1, 30, 2},
	{0xfb, -57344, 1, 30, 3},
	{0xfc, float32(math.Inf(-1)), 1, 31, 0},
	{0xfd, float32(math.NaN()), 1, 31, 1},
	{0xfe, float32(math.NaN()), 1, 31, 2},
	{0xff, float32(math.NaN()), 1, 31, 3},
}
