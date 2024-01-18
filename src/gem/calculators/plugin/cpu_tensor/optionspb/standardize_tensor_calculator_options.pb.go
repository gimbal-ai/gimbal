// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/calculators/plugin/cpu_tensor/optionspb/standardize_tensor_calculator_options.proto

package optionspb

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	io "io"
	math "math"
	math_bits "math/bits"
	reflect "reflect"
	strings "strings"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.GoGoProtoPackageIsVersion3 // please upgrade the proto package

type StandardizeTensorCalculatorOptions struct {
	Mean   []float32 `protobuf:"fixed32,1,rep,packed,name=mean,proto3" json:"mean,omitempty"`
	Stddev []float32 `protobuf:"fixed32,2,rep,packed,name=stddev,proto3" json:"stddev,omitempty"`
}

func (m *StandardizeTensorCalculatorOptions) Reset()      { *m = StandardizeTensorCalculatorOptions{} }
func (*StandardizeTensorCalculatorOptions) ProtoMessage() {}
func (*StandardizeTensorCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_512189483479c0ba, []int{0}
}
func (m *StandardizeTensorCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *StandardizeTensorCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_StandardizeTensorCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *StandardizeTensorCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_StandardizeTensorCalculatorOptions.Merge(m, src)
}
func (m *StandardizeTensorCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *StandardizeTensorCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_StandardizeTensorCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_StandardizeTensorCalculatorOptions proto.InternalMessageInfo

func (m *StandardizeTensorCalculatorOptions) GetMean() []float32 {
	if m != nil {
		return m.Mean
	}
	return nil
}

func (m *StandardizeTensorCalculatorOptions) GetStddev() []float32 {
	if m != nil {
		return m.Stddev
	}
	return nil
}

func init() {
	proto.RegisterType((*StandardizeTensorCalculatorOptions)(nil), "gml.gem.calculators.cpu_tensor.optionspb.StandardizeTensorCalculatorOptions")
}

func init() {
	proto.RegisterFile("src/gem/calculators/plugin/cpu_tensor/optionspb/standardize_tensor_calculator_options.proto", fileDescriptor_512189483479c0ba)
}

var fileDescriptor_512189483479c0ba = []byte{
	// 251 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x90, 0xad, 0x4e, 0xc4, 0x40,
	0x14, 0x85, 0xe7, 0x2e, 0x64, 0x45, 0x65, 0x05, 0x59, 0x75, 0x43, 0x56, 0xad, 0x9a, 0x11, 0x48,
	0x1c, 0x78, 0x20, 0x80, 0x02, 0xd1, 0x4c, 0xdb, 0x49, 0x33, 0xc9, 0xfc, 0x65, 0x66, 0x8a, 0x40,
	0x21, 0x78, 0x00, 0x1e, 0x83, 0x47, 0x41, 0x56, 0xae, 0xa4, 0x53, 0x83, 0xdc, 0x47, 0x20, 0xe9,
	0x36, 0x2d, 0x16, 0x77, 0xcf, 0x3d, 0xe7, 0x7c, 0xe2, 0x64, 0xcf, 0xc1, 0x57, 0xac, 0x11, 0x9a,
	0x55, 0x5c, 0x55, 0xad, 0xe2, 0xd1, 0xfa, 0xc0, 0x9c, 0x6a, 0x1b, 0x69, 0x58, 0xe5, 0xda, 0x22,
	0x0a, 0x13, 0xac, 0x67, 0xd6, 0x45, 0x69, 0x4d, 0x70, 0x25, 0x0b, 0x91, 0x9b, 0x9a, 0xfb, 0x5a,
	0xbe, 0x8a, 0xc9, 0x2c, 0x96, 0x6a, 0x31, 0xe5, 0xa8, 0xf3, 0x36, 0xda, 0x7c, 0xd7, 0x68, 0x45,
	0x1b, 0xa1, 0xe9, 0x1f, 0x38, 0x5d, 0xa8, 0x74, 0xa6, 0x6e, 0xef, 0xb2, 0xed, 0xc3, 0x02, 0x7e,
	0x1c, 0xed, 0xeb, 0xb9, 0x74, 0x7b, 0xcc, 0xe5, 0x79, 0x76, 0xaa, 0x05, 0x37, 0x1b, 0x38, 0x3f,
	0xd9, 0xad, 0xee, 0xc7, 0x3b, 0x3f, 0xcb, 0xd6, 0x21, 0xd6, 0xb5, 0x78, 0xd9, 0xac, 0xc6, 0xef,
	0xa4, 0xae, 0xde, 0xa1, 0xeb, 0x91, 0xec, 0x7b, 0x24, 0x87, 0x1e, 0xe1, 0x2d, 0x21, 0x7c, 0x26,
	0x84, 0xaf, 0x84, 0xd0, 0x25, 0x84, 0xef, 0x84, 0xf0, 0x93, 0x90, 0x1c, 0x12, 0xc2, 0xc7, 0x80,
	0xa4, 0x1b, 0x90, 0xec, 0x07, 0x24, 0x4f, 0x37, 0x8d, 0xd4, 0x4a, 0x44, 0xc5, 0xcb, 0x40, 0xb9,
	0x64, 0x47, 0xc5, 0xfe, 0x39, 0xd2, 0xe5, 0x7c, 0x95, 0xeb, 0x71, 0x89, 0x8b, 0xdf, 0x00, 0x00,
	0x00, 0xff, 0xff, 0x6c, 0x56, 0xaa, 0x3d, 0x68, 0x01, 0x00, 0x00,
}

func (this *StandardizeTensorCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*StandardizeTensorCalculatorOptions)
	if !ok {
		that2, ok := that.(StandardizeTensorCalculatorOptions)
		if ok {
			that1 = &that2
		} else {
			return false
		}
	}
	if that1 == nil {
		return this == nil
	} else if this == nil {
		return false
	}
	if len(this.Mean) != len(that1.Mean) {
		return false
	}
	for i := range this.Mean {
		if this.Mean[i] != that1.Mean[i] {
			return false
		}
	}
	if len(this.Stddev) != len(that1.Stddev) {
		return false
	}
	for i := range this.Stddev {
		if this.Stddev[i] != that1.Stddev[i] {
			return false
		}
	}
	return true
}
func (this *StandardizeTensorCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&optionspb.StandardizeTensorCalculatorOptions{")
	s = append(s, "Mean: "+fmt.Sprintf("%#v", this.Mean)+",\n")
	s = append(s, "Stddev: "+fmt.Sprintf("%#v", this.Stddev)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringStandardizeTensorCalculatorOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *StandardizeTensorCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *StandardizeTensorCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *StandardizeTensorCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Stddev) > 0 {
		for iNdEx := len(m.Stddev) - 1; iNdEx >= 0; iNdEx-- {
			f1 := math.Float32bits(float32(m.Stddev[iNdEx]))
			i -= 4
			encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(f1))
		}
		i = encodeVarintStandardizeTensorCalculatorOptions(dAtA, i, uint64(len(m.Stddev)*4))
		i--
		dAtA[i] = 0x12
	}
	if len(m.Mean) > 0 {
		for iNdEx := len(m.Mean) - 1; iNdEx >= 0; iNdEx-- {
			f2 := math.Float32bits(float32(m.Mean[iNdEx]))
			i -= 4
			encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(f2))
		}
		i = encodeVarintStandardizeTensorCalculatorOptions(dAtA, i, uint64(len(m.Mean)*4))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintStandardizeTensorCalculatorOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovStandardizeTensorCalculatorOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *StandardizeTensorCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Mean) > 0 {
		n += 1 + sovStandardizeTensorCalculatorOptions(uint64(len(m.Mean)*4)) + len(m.Mean)*4
	}
	if len(m.Stddev) > 0 {
		n += 1 + sovStandardizeTensorCalculatorOptions(uint64(len(m.Stddev)*4)) + len(m.Stddev)*4
	}
	return n
}

func sovStandardizeTensorCalculatorOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozStandardizeTensorCalculatorOptions(x uint64) (n int) {
	return sovStandardizeTensorCalculatorOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *StandardizeTensorCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&StandardizeTensorCalculatorOptions{`,
		`Mean:` + fmt.Sprintf("%v", this.Mean) + `,`,
		`Stddev:` + fmt.Sprintf("%v", this.Stddev) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringStandardizeTensorCalculatorOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *StandardizeTensorCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowStandardizeTensorCalculatorOptions
			}
			if iNdEx >= l {
				return io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= uint64(b&0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		fieldNum := int32(wire >> 3)
		wireType := int(wire & 0x7)
		if wireType == 4 {
			return fmt.Errorf("proto: StandardizeTensorCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: StandardizeTensorCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType == 5 {
				var v uint32
				if (iNdEx + 4) > l {
					return io.ErrUnexpectedEOF
				}
				v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
				iNdEx += 4
				v2 := float32(math.Float32frombits(v))
				m.Mean = append(m.Mean, v2)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowStandardizeTensorCalculatorOptions
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthStandardizeTensorCalculatorOptions
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthStandardizeTensorCalculatorOptions
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				elementCount = packedLen / 4
				if elementCount != 0 && len(m.Mean) == 0 {
					m.Mean = make([]float32, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v uint32
					if (iNdEx + 4) > l {
						return io.ErrUnexpectedEOF
					}
					v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
					iNdEx += 4
					v2 := float32(math.Float32frombits(v))
					m.Mean = append(m.Mean, v2)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field Mean", wireType)
			}
		case 2:
			if wireType == 5 {
				var v uint32
				if (iNdEx + 4) > l {
					return io.ErrUnexpectedEOF
				}
				v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
				iNdEx += 4
				v2 := float32(math.Float32frombits(v))
				m.Stddev = append(m.Stddev, v2)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowStandardizeTensorCalculatorOptions
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthStandardizeTensorCalculatorOptions
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthStandardizeTensorCalculatorOptions
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				elementCount = packedLen / 4
				if elementCount != 0 && len(m.Stddev) == 0 {
					m.Stddev = make([]float32, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v uint32
					if (iNdEx + 4) > l {
						return io.ErrUnexpectedEOF
					}
					v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
					iNdEx += 4
					v2 := float32(math.Float32frombits(v))
					m.Stddev = append(m.Stddev, v2)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field Stddev", wireType)
			}
		default:
			iNdEx = preIndex
			skippy, err := skipStandardizeTensorCalculatorOptions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthStandardizeTensorCalculatorOptions
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipStandardizeTensorCalculatorOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowStandardizeTensorCalculatorOptions
			}
			if iNdEx >= l {
				return 0, io.ErrUnexpectedEOF
			}
			b := dAtA[iNdEx]
			iNdEx++
			wire |= (uint64(b) & 0x7F) << shift
			if b < 0x80 {
				break
			}
		}
		wireType := int(wire & 0x7)
		switch wireType {
		case 0:
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowStandardizeTensorCalculatorOptions
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				iNdEx++
				if dAtA[iNdEx-1] < 0x80 {
					break
				}
			}
		case 1:
			iNdEx += 8
		case 2:
			var length int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return 0, ErrIntOverflowStandardizeTensorCalculatorOptions
				}
				if iNdEx >= l {
					return 0, io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				length |= (int(b) & 0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if length < 0 {
				return 0, ErrInvalidLengthStandardizeTensorCalculatorOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupStandardizeTensorCalculatorOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthStandardizeTensorCalculatorOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthStandardizeTensorCalculatorOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowStandardizeTensorCalculatorOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupStandardizeTensorCalculatorOptions = fmt.Errorf("proto: unexpected end of group")
)
