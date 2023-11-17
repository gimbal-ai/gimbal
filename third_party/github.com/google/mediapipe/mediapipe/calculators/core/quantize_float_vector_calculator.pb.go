// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/core/quantize_float_vector_calculator.proto

package core

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	framework "github.com/google/mediapipe/mediapipe/framework"
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

type QuantizeFloatVectorCalculatorOptions struct {
	MaxQuantizedValue float32 `protobuf:"fixed32,1,opt,name=max_quantized_value,json=maxQuantizedValue" json:"max_quantized_value"`
	MinQuantizedValue float32 `protobuf:"fixed32,2,opt,name=min_quantized_value,json=minQuantizedValue" json:"min_quantized_value"`
}

func (m *QuantizeFloatVectorCalculatorOptions) Reset()      { *m = QuantizeFloatVectorCalculatorOptions{} }
func (*QuantizeFloatVectorCalculatorOptions) ProtoMessage() {}
func (*QuantizeFloatVectorCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_82e80fedce49dc76, []int{0}
}
func (m *QuantizeFloatVectorCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *QuantizeFloatVectorCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_QuantizeFloatVectorCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *QuantizeFloatVectorCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_QuantizeFloatVectorCalculatorOptions.Merge(m, src)
}
func (m *QuantizeFloatVectorCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *QuantizeFloatVectorCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_QuantizeFloatVectorCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_QuantizeFloatVectorCalculatorOptions proto.InternalMessageInfo

func (m *QuantizeFloatVectorCalculatorOptions) GetMaxQuantizedValue() float32 {
	if m != nil {
		return m.MaxQuantizedValue
	}
	return 0
}

func (m *QuantizeFloatVectorCalculatorOptions) GetMinQuantizedValue() float32 {
	if m != nil {
		return m.MinQuantizedValue
	}
	return 0
}

var E_QuantizeFloatVectorCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*QuantizeFloatVectorCalculatorOptions)(nil),
	Field:         259848061,
	Name:          "mediapipe.QuantizeFloatVectorCalculatorOptions.ext",
	Tag:           "bytes,259848061,opt,name=ext",
	Filename:      "mediapipe/calculators/core/quantize_float_vector_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_QuantizeFloatVectorCalculatorOptions_Ext)
	proto.RegisterType((*QuantizeFloatVectorCalculatorOptions)(nil), "mediapipe.QuantizeFloatVectorCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/core/quantize_float_vector_calculator.proto", fileDescriptor_82e80fedce49dc76)
}

var fileDescriptor_82e80fedce49dc76 = []byte{
	// 299 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x72, 0xcc, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0x4f, 0xce, 0x2f, 0x4a, 0xd5, 0x2f, 0x2c, 0x4d, 0xcc, 0x2b, 0xc9, 0xac, 0x4a, 0x8d,
	0x4f, 0xcb, 0xc9, 0x4f, 0x2c, 0x89, 0x2f, 0x4b, 0x4d, 0x2e, 0xc9, 0x2f, 0x8a, 0x47, 0x28, 0xd3,
	0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0x84, 0x1b, 0x21, 0xa5, 0x82, 0x30, 0x2d, 0xad, 0x28,
	0x31, 0x37, 0xb5, 0x3c, 0xbf, 0x28, 0x5b, 0x1f, 0x5d, 0x83, 0xd2, 0x37, 0x46, 0x2e, 0x95, 0x40,
	0xa8, 0xd9, 0x6e, 0x20, 0xa3, 0xc3, 0xc0, 0x26, 0x3b, 0xc3, 0xd5, 0xf9, 0x17, 0x94, 0x64, 0xe6,
	0xe7, 0x15, 0x0b, 0x99, 0x70, 0x09, 0xe7, 0x26, 0x56, 0xc4, 0xc3, 0xdc, 0x91, 0x12, 0x5f, 0x96,
	0x98, 0x53, 0x9a, 0x2a, 0xc1, 0xa8, 0xc0, 0xa8, 0xc1, 0xe4, 0xc4, 0x72, 0xe2, 0x9e, 0x3c, 0x43,
	0x90, 0x60, 0x6e, 0x62, 0x05, 0xcc, 0xac, 0x94, 0x30, 0x90, 0x34, 0x58, 0x57, 0x66, 0x1e, 0x86,
	0x2e, 0x26, 0x14, 0x5d, 0x99, 0x79, 0xa8, 0xba, 0x8c, 0x92, 0xb8, 0x98, 0x53, 0x2b, 0x4a, 0x84,
	0x64, 0xf4, 0xe0, 0x5e, 0xd0, 0xc3, 0x70, 0x90, 0xc4, 0xdf, 0x77, 0x9f, 0xab, 0x15, 0x18, 0x35,
	0xb8, 0x8d, 0xf4, 0x91, 0x94, 0x11, 0xe3, 0x95, 0x20, 0x90, 0xe1, 0x4e, 0x39, 0x17, 0x1e, 0xca,
	0x31, 0xdc, 0x78, 0x28, 0xc7, 0xf0, 0xe1, 0xa1, 0x1c, 0x63, 0xc3, 0x23, 0x39, 0xc6, 0x15, 0x8f,
	0xe4, 0x18, 0x4f, 0x3c, 0x92, 0x63, 0xbc, 0xf0, 0x48, 0x8e, 0xf1, 0xc1, 0x23, 0x39, 0xc6, 0x17,
	0x8f, 0xe4, 0x18, 0x3e, 0x3c, 0x92, 0x63, 0x9c, 0xf0, 0x58, 0x8e, 0xe1, 0xc2, 0x63, 0x39, 0x86,
	0x1b, 0x8f, 0xe5, 0x18, 0xa2, 0xcc, 0xd2, 0x33, 0x4b, 0x32, 0x4a, 0x93, 0xf4, 0x92, 0xf3, 0x73,
	0xf5, 0xd3, 0xf3, 0xf3, 0xd3, 0x73, 0x52, 0xf5, 0x11, 0xc1, 0x8c, 0x3b, 0xfa, 0x00, 0x01, 0x00,
	0x00, 0xff, 0xff, 0xdb, 0xe6, 0xcc, 0x0c, 0xdb, 0x01, 0x00, 0x00,
}

func (this *QuantizeFloatVectorCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*QuantizeFloatVectorCalculatorOptions)
	if !ok {
		that2, ok := that.(QuantizeFloatVectorCalculatorOptions)
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
	if this.MaxQuantizedValue != that1.MaxQuantizedValue {
		return false
	}
	if this.MinQuantizedValue != that1.MinQuantizedValue {
		return false
	}
	return true
}
func (this *QuantizeFloatVectorCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&core.QuantizeFloatVectorCalculatorOptions{")
	s = append(s, "MaxQuantizedValue: "+fmt.Sprintf("%#v", this.MaxQuantizedValue)+",\n")
	s = append(s, "MinQuantizedValue: "+fmt.Sprintf("%#v", this.MinQuantizedValue)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringQuantizeFloatVectorCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *QuantizeFloatVectorCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *QuantizeFloatVectorCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *QuantizeFloatVectorCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.MinQuantizedValue))))
	i--
	dAtA[i] = 0x15
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.MaxQuantizedValue))))
	i--
	dAtA[i] = 0xd
	return len(dAtA) - i, nil
}

func encodeVarintQuantizeFloatVectorCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovQuantizeFloatVectorCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *QuantizeFloatVectorCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 5
	n += 5
	return n
}

func sovQuantizeFloatVectorCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozQuantizeFloatVectorCalculator(x uint64) (n int) {
	return sovQuantizeFloatVectorCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *QuantizeFloatVectorCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&QuantizeFloatVectorCalculatorOptions{`,
		`MaxQuantizedValue:` + fmt.Sprintf("%v", this.MaxQuantizedValue) + `,`,
		`MinQuantizedValue:` + fmt.Sprintf("%v", this.MinQuantizedValue) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringQuantizeFloatVectorCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *QuantizeFloatVectorCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowQuantizeFloatVectorCalculator
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
			return fmt.Errorf("proto: QuantizeFloatVectorCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: QuantizeFloatVectorCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field MaxQuantizedValue", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.MaxQuantizedValue = float32(math.Float32frombits(v))
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field MinQuantizedValue", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.MinQuantizedValue = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipQuantizeFloatVectorCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthQuantizeFloatVectorCalculator
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
func skipQuantizeFloatVectorCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowQuantizeFloatVectorCalculator
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
					return 0, ErrIntOverflowQuantizeFloatVectorCalculator
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
					return 0, ErrIntOverflowQuantizeFloatVectorCalculator
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
				return 0, ErrInvalidLengthQuantizeFloatVectorCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupQuantizeFloatVectorCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthQuantizeFloatVectorCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthQuantizeFloatVectorCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowQuantizeFloatVectorCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupQuantizeFloatVectorCalculator = fmt.Errorf("proto: unexpected end of group")
)