// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/tensor/tensors_to_floats_calculator.proto

package tensor

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	framework "github.com/google/mediapipe/mediapipe/framework"
	io "io"
	math "math"
	math_bits "math/bits"
	reflect "reflect"
	strconv "strconv"
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

type TensorsToFloatsCalculatorOptions_Activation int32

const (
	T2F_ACTIVATION_NONE    TensorsToFloatsCalculatorOptions_Activation = 0
	T2F_ACTIVATION_SIGMOID TensorsToFloatsCalculatorOptions_Activation = 1
)

var TensorsToFloatsCalculatorOptions_Activation_name = map[int32]string{
	0: "T2F_ACTIVATION_NONE",
	1: "T2F_ACTIVATION_SIGMOID",
}

var TensorsToFloatsCalculatorOptions_Activation_value = map[string]int32{
	"T2F_ACTIVATION_NONE":    0,
	"T2F_ACTIVATION_SIGMOID": 1,
}

func (x TensorsToFloatsCalculatorOptions_Activation) Enum() *TensorsToFloatsCalculatorOptions_Activation {
	p := new(TensorsToFloatsCalculatorOptions_Activation)
	*p = x
	return p
}

func (x TensorsToFloatsCalculatorOptions_Activation) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(TensorsToFloatsCalculatorOptions_Activation_name, int32(x))
}

func (x *TensorsToFloatsCalculatorOptions_Activation) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(TensorsToFloatsCalculatorOptions_Activation_value, data, "TensorsToFloatsCalculatorOptions_Activation")
	if err != nil {
		return err
	}
	*x = TensorsToFloatsCalculatorOptions_Activation(value)
	return nil
}

func (TensorsToFloatsCalculatorOptions_Activation) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_7959324debe50431, []int{0, 0}
}

type TensorsToFloatsCalculatorOptions struct {
	Activation *TensorsToFloatsCalculatorOptions_Activation `protobuf:"varint,1,opt,name=activation,enum=mediapipe.TensorsToFloatsCalculatorOptions_Activation,def=0" json:"activation,omitempty"`
}

func (m *TensorsToFloatsCalculatorOptions) Reset()      { *m = TensorsToFloatsCalculatorOptions{} }
func (*TensorsToFloatsCalculatorOptions) ProtoMessage() {}
func (*TensorsToFloatsCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_7959324debe50431, []int{0}
}
func (m *TensorsToFloatsCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TensorsToFloatsCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TensorsToFloatsCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TensorsToFloatsCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TensorsToFloatsCalculatorOptions.Merge(m, src)
}
func (m *TensorsToFloatsCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *TensorsToFloatsCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_TensorsToFloatsCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_TensorsToFloatsCalculatorOptions proto.InternalMessageInfo

const Default_TensorsToFloatsCalculatorOptions_Activation TensorsToFloatsCalculatorOptions_Activation = T2F_ACTIVATION_NONE

func (m *TensorsToFloatsCalculatorOptions) GetActivation() TensorsToFloatsCalculatorOptions_Activation {
	if m != nil && m.Activation != nil {
		return *m.Activation
	}
	return Default_TensorsToFloatsCalculatorOptions_Activation
}

var E_TensorsToFloatsCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*TensorsToFloatsCalculatorOptions)(nil),
	Field:         343499115,
	Name:          "mediapipe.TensorsToFloatsCalculatorOptions.ext",
	Tag:           "bytes,343499115,opt,name=ext",
	Filename:      "mediapipe/calculators/tensor/tensors_to_floats_calculator.proto",
}

func init() {
	proto.RegisterEnum("mediapipe.TensorsToFloatsCalculatorOptions_Activation", TensorsToFloatsCalculatorOptions_Activation_name, TensorsToFloatsCalculatorOptions_Activation_value)
	proto.RegisterExtension(E_TensorsToFloatsCalculatorOptions_Ext)
	proto.RegisterType((*TensorsToFloatsCalculatorOptions)(nil), "mediapipe.TensorsToFloatsCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/tensor/tensors_to_floats_calculator.proto", fileDescriptor_7959324debe50431)
}

var fileDescriptor_7959324debe50431 = []byte{
	// 315 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xb2, 0xcf, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0x2f, 0x49, 0xcd, 0x2b, 0xce, 0x2f, 0x82, 0x52, 0xc5, 0xf1, 0x25, 0xf9, 0xf1, 0x69,
	0x39, 0xf9, 0x89, 0x25, 0xc5, 0xf1, 0x08, 0x45, 0x7a, 0x05, 0x45, 0xf9, 0x25, 0xf9, 0x42, 0x9c,
	0x70, 0x03, 0xa4, 0x54, 0x10, 0x66, 0xa5, 0x15, 0x25, 0xe6, 0xa6, 0x96, 0xe7, 0x17, 0x65, 0xeb,
	0xa3, 0x6b, 0x50, 0xda, 0xcc, 0xc4, 0xa5, 0x10, 0x02, 0x31, 0x37, 0x24, 0xdf, 0x0d, 0x6c, 0xaa,
	0x33, 0x5c, 0x8d, 0x7f, 0x41, 0x49, 0x66, 0x7e, 0x5e, 0xb1, 0x50, 0x36, 0x17, 0x57, 0x62, 0x72,
	0x49, 0x66, 0x59, 0x22, 0x88, 0x2b, 0xc1, 0xa8, 0xc0, 0xa8, 0xc1, 0x67, 0x64, 0xa6, 0x07, 0x37,
	0x5f, 0x8f, 0x90, 0x01, 0x7a, 0x8e, 0x70, 0xdd, 0x56, 0xc2, 0x21, 0x46, 0x6e, 0xf1, 0x8e, 0xce,
	0x21, 0x9e, 0x61, 0x8e, 0x21, 0x9e, 0xfe, 0x7e, 0xf1, 0x7e, 0xfe, 0x7e, 0xae, 0x41, 0x48, 0xc6,
	0x2b, 0x39, 0x72, 0x71, 0x21, 0x94, 0x0b, 0x89, 0x73, 0x61, 0xd3, 0x20, 0xc0, 0x20, 0x24, 0xc5,
	0x25, 0x86, 0x26, 0x11, 0xec, 0xe9, 0xee, 0xeb, 0xef, 0xe9, 0x22, 0xc0, 0x68, 0x14, 0xcf, 0xc5,
	0x9c, 0x5a, 0x51, 0x22, 0x24, 0x83, 0xe4, 0x44, 0x0c, 0x37, 0x49, 0xbc, 0x3e, 0xf4, 0x74, 0x31,
	0xc8, 0x2b, 0xdc, 0x46, 0xda, 0x24, 0x78, 0x25, 0x08, 0x64, 0xb2, 0x53, 0xde, 0x85, 0x87, 0x72,
	0x0c, 0x37, 0x1e, 0xca, 0x31, 0x7c, 0x78, 0x28, 0xc7, 0xd8, 0xf0, 0x48, 0x8e, 0x71, 0xc5, 0x23,
	0x39, 0xc6, 0x13, 0x8f, 0xe4, 0x18, 0x2f, 0x3c, 0x92, 0x63, 0x7c, 0xf0, 0x48, 0x8e, 0xf1, 0xc5,
	0x23, 0x39, 0x86, 0x0f, 0x8f, 0xe4, 0x18, 0x27, 0x3c, 0x96, 0x63, 0xb8, 0xf0, 0x58, 0x8e, 0xe1,
	0xc6, 0x63, 0x39, 0x86, 0x28, 0x8b, 0xf4, 0xcc, 0x92, 0x8c, 0xd2, 0x24, 0xbd, 0xe4, 0xfc, 0x5c,
	0xfd, 0xf4, 0xfc, 0xfc, 0xf4, 0x9c, 0x54, 0x7d, 0x44, 0x1c, 0xe1, 0x8b, 0x79, 0x40, 0x00, 0x00,
	0x00, 0xff, 0xff, 0x52, 0x44, 0xe1, 0x87, 0x18, 0x02, 0x00, 0x00,
}

func (x TensorsToFloatsCalculatorOptions_Activation) String() string {
	s, ok := TensorsToFloatsCalculatorOptions_Activation_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *TensorsToFloatsCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*TensorsToFloatsCalculatorOptions)
	if !ok {
		that2, ok := that.(TensorsToFloatsCalculatorOptions)
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
	if this.Activation != nil && that1.Activation != nil {
		if *this.Activation != *that1.Activation {
			return false
		}
	} else if this.Activation != nil {
		return false
	} else if that1.Activation != nil {
		return false
	}
	return true
}
func (this *TensorsToFloatsCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&tensor.TensorsToFloatsCalculatorOptions{")
	if this.Activation != nil {
		s = append(s, "Activation: "+valueToGoStringTensorsToFloatsCalculator(this.Activation, "TensorsToFloatsCalculatorOptions_Activation")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringTensorsToFloatsCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *TensorsToFloatsCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TensorsToFloatsCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *TensorsToFloatsCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Activation != nil {
		i = encodeVarintTensorsToFloatsCalculator(dAtA, i, uint64(*m.Activation))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintTensorsToFloatsCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovTensorsToFloatsCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *TensorsToFloatsCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Activation != nil {
		n += 1 + sovTensorsToFloatsCalculator(uint64(*m.Activation))
	}
	return n
}

func sovTensorsToFloatsCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozTensorsToFloatsCalculator(x uint64) (n int) {
	return sovTensorsToFloatsCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *TensorsToFloatsCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&TensorsToFloatsCalculatorOptions{`,
		`Activation:` + valueToStringTensorsToFloatsCalculator(this.Activation) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringTensorsToFloatsCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *TensorsToFloatsCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorsToFloatsCalculator
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
			return fmt.Errorf("proto: TensorsToFloatsCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TensorsToFloatsCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Activation", wireType)
			}
			var v TensorsToFloatsCalculatorOptions_Activation
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorsToFloatsCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= TensorsToFloatsCalculatorOptions_Activation(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.Activation = &v
		default:
			iNdEx = preIndex
			skippy, err := skipTensorsToFloatsCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTensorsToFloatsCalculator
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
func skipTensorsToFloatsCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTensorsToFloatsCalculator
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
					return 0, ErrIntOverflowTensorsToFloatsCalculator
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
					return 0, ErrIntOverflowTensorsToFloatsCalculator
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
				return 0, ErrInvalidLengthTensorsToFloatsCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupTensorsToFloatsCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthTensorsToFloatsCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthTensorsToFloatsCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTensorsToFloatsCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupTensorsToFloatsCalculator = fmt.Errorf("proto: unexpected end of group")
)
