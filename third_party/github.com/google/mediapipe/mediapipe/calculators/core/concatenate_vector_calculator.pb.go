// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/core/concatenate_vector_calculator.proto

package core

import (
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

type ConcatenateVectorCalculatorOptions struct {
	OnlyEmitIfAllPresent *bool `protobuf:"varint,1,opt,name=only_emit_if_all_present,json=onlyEmitIfAllPresent,def=0" json:"only_emit_if_all_present,omitempty"`
}

func (m *ConcatenateVectorCalculatorOptions) Reset()      { *m = ConcatenateVectorCalculatorOptions{} }
func (*ConcatenateVectorCalculatorOptions) ProtoMessage() {}
func (*ConcatenateVectorCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_897d802642a37321, []int{0}
}
func (m *ConcatenateVectorCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ConcatenateVectorCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ConcatenateVectorCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ConcatenateVectorCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ConcatenateVectorCalculatorOptions.Merge(m, src)
}
func (m *ConcatenateVectorCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *ConcatenateVectorCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_ConcatenateVectorCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_ConcatenateVectorCalculatorOptions proto.InternalMessageInfo

const Default_ConcatenateVectorCalculatorOptions_OnlyEmitIfAllPresent bool = false

func (m *ConcatenateVectorCalculatorOptions) GetOnlyEmitIfAllPresent() bool {
	if m != nil && m.OnlyEmitIfAllPresent != nil {
		return *m.OnlyEmitIfAllPresent
	}
	return Default_ConcatenateVectorCalculatorOptions_OnlyEmitIfAllPresent
}

var E_ConcatenateVectorCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*ConcatenateVectorCalculatorOptions)(nil),
	Field:         259397839,
	Name:          "mediapipe.ConcatenateVectorCalculatorOptions.ext",
	Tag:           "bytes,259397839,opt,name=ext",
	Filename:      "mediapipe/calculators/core/concatenate_vector_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_ConcatenateVectorCalculatorOptions_Ext)
	proto.RegisterType((*ConcatenateVectorCalculatorOptions)(nil), "mediapipe.ConcatenateVectorCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/core/concatenate_vector_calculator.proto", fileDescriptor_897d802642a37321)
}

var fileDescriptor_897d802642a37321 = []byte{
	// 295 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x91, 0xbf, 0x4a, 0x33, 0x41,
	0x14, 0xc5, 0x67, 0xf8, 0xf8, 0x40, 0xd7, 0x2e, 0x58, 0x04, 0x91, 0x4b, 0x08, 0x16, 0x69, 0xdc,
	0x81, 0x14, 0x16, 0x82, 0x82, 0x06, 0x0b, 0x2b, 0x25, 0x85, 0x85, 0xcd, 0x38, 0x8e, 0x77, 0xe3,
	0xe0, 0xec, 0xde, 0x65, 0x76, 0xe2, 0x1f, 0x6c, 0x7c, 0x04, 0x1f, 0xc3, 0xd6, 0x87, 0x10, 0xec,
	0x4c, 0x99, 0xd2, 0x9d, 0x6d, 0x2c, 0xf3, 0x08, 0x12, 0x85, 0x5d, 0x51, 0xc4, 0xfe, 0x77, 0x7e,
	0xf7, 0x70, 0x4f, 0xb4, 0x9d, 0xe2, 0x99, 0x51, 0xb9, 0xc9, 0x51, 0x68, 0x65, 0xf5, 0xd8, 0x2a,
	0x4f, 0xae, 0x10, 0x9a, 0x1c, 0x0a, 0x4d, 0x99, 0x56, 0x1e, 0x33, 0xe5, 0x51, 0x5e, 0xa2, 0xf6,
	0xe4, 0x64, 0xc3, 0xc4, 0xb9, 0x23, 0x4f, 0xad, 0xc5, 0x3a, 0xbf, 0xb2, 0xd6, 0xa8, 0x12, 0xa7,
	0x52, 0xbc, 0x22, 0x77, 0x21, 0xbe, 0x07, 0xba, 0x4f, 0x3c, 0xea, 0x0e, 0x1a, 0xf1, 0xd1, 0x87,
	0x77, 0x50, 0x53, 0x07, 0xb9, 0x37, 0x94, 0x15, 0xad, 0xad, 0xa8, 0x4d, 0x99, 0xbd, 0x91, 0x98,
	0x1a, 0x2f, 0x4d, 0x22, 0x95, 0xb5, 0x32, 0x77, 0x58, 0x60, 0xe6, 0xdb, 0xbc, 0xc3, 0x7b, 0x0b,
	0x9b, 0xff, 0x13, 0x65, 0x0b, 0x1c, 0x2e, 0xcf, 0xb1, 0xbd, 0xd4, 0xf8, 0xfd, 0x64, 0xc7, 0xda,
	0xc3, 0x4f, 0xa4, 0x7f, 0x12, 0xfd, 0xc3, 0x6b, 0xdf, 0x5a, 0x8d, 0xeb, 0x4e, 0xf1, 0x8f, 0x1b,
	0xed, 0x97, 0xc7, 0xe9, 0x6d, 0x87, 0xf7, 0x96, 0xfa, 0xeb, 0x5f, 0xb1, 0x3f, 0xbb, 0x0d, 0xe7,
	0xea, 0x5d, 0x3b, 0x29, 0x81, 0x4d, 0x4b, 0x60, 0xb3, 0x12, 0xf8, 0x5d, 0x00, 0xfe, 0x10, 0x80,
	0x3f, 0x07, 0xe0, 0x93, 0x00, 0xfc, 0x35, 0x00, 0x7f, 0x0b, 0xc0, 0x66, 0x01, 0xf8, 0x7d, 0x05,
	0x6c, 0x52, 0x01, 0x9b, 0x56, 0xc0, 0x8e, 0x37, 0x46, 0xc6, 0x9f, 0x8f, 0x4f, 0x63, 0x4d, 0xa9,
	0x18, 0x11, 0x8d, 0x2c, 0x8a, 0xe6, 0x6b, 0xbf, 0x4f, 0xf1, 0x1e, 0x00, 0x00, 0xff, 0xff, 0xd2,
	0x6b, 0x49, 0x7e, 0xa7, 0x01, 0x00, 0x00,
}

func (this *ConcatenateVectorCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ConcatenateVectorCalculatorOptions)
	if !ok {
		that2, ok := that.(ConcatenateVectorCalculatorOptions)
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
	if this.OnlyEmitIfAllPresent != nil && that1.OnlyEmitIfAllPresent != nil {
		if *this.OnlyEmitIfAllPresent != *that1.OnlyEmitIfAllPresent {
			return false
		}
	} else if this.OnlyEmitIfAllPresent != nil {
		return false
	} else if that1.OnlyEmitIfAllPresent != nil {
		return false
	}
	return true
}
func (this *ConcatenateVectorCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&core.ConcatenateVectorCalculatorOptions{")
	if this.OnlyEmitIfAllPresent != nil {
		s = append(s, "OnlyEmitIfAllPresent: "+valueToGoStringConcatenateVectorCalculator(this.OnlyEmitIfAllPresent, "bool")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringConcatenateVectorCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *ConcatenateVectorCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ConcatenateVectorCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ConcatenateVectorCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.OnlyEmitIfAllPresent != nil {
		i--
		if *m.OnlyEmitIfAllPresent {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintConcatenateVectorCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovConcatenateVectorCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ConcatenateVectorCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.OnlyEmitIfAllPresent != nil {
		n += 2
	}
	return n
}

func sovConcatenateVectorCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozConcatenateVectorCalculator(x uint64) (n int) {
	return sovConcatenateVectorCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ConcatenateVectorCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ConcatenateVectorCalculatorOptions{`,
		`OnlyEmitIfAllPresent:` + valueToStringConcatenateVectorCalculator(this.OnlyEmitIfAllPresent) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringConcatenateVectorCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ConcatenateVectorCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowConcatenateVectorCalculator
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
			return fmt.Errorf("proto: ConcatenateVectorCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ConcatenateVectorCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OnlyEmitIfAllPresent", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowConcatenateVectorCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			b := bool(v != 0)
			m.OnlyEmitIfAllPresent = &b
		default:
			iNdEx = preIndex
			skippy, err := skipConcatenateVectorCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthConcatenateVectorCalculator
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
func skipConcatenateVectorCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowConcatenateVectorCalculator
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
					return 0, ErrIntOverflowConcatenateVectorCalculator
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
					return 0, ErrIntOverflowConcatenateVectorCalculator
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
				return 0, ErrInvalidLengthConcatenateVectorCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupConcatenateVectorCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthConcatenateVectorCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthConcatenateVectorCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowConcatenateVectorCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupConcatenateVectorCalculator = fmt.Errorf("proto: unexpected end of group")
)
