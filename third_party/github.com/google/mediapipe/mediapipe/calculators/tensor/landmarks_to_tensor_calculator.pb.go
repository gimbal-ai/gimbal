// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/tensor/landmarks_to_tensor_calculator.proto

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

type LandmarksToTensorCalculatorOptions_Attribute int32

const (
	X          LandmarksToTensorCalculatorOptions_Attribute = 0
	Y          LandmarksToTensorCalculatorOptions_Attribute = 1
	Z          LandmarksToTensorCalculatorOptions_Attribute = 2
	VISIBILITY LandmarksToTensorCalculatorOptions_Attribute = 3
	PRESENCE   LandmarksToTensorCalculatorOptions_Attribute = 4
)

var LandmarksToTensorCalculatorOptions_Attribute_name = map[int32]string{
	0: "X",
	1: "Y",
	2: "Z",
	3: "VISIBILITY",
	4: "PRESENCE",
}

var LandmarksToTensorCalculatorOptions_Attribute_value = map[string]int32{
	"X":          0,
	"Y":          1,
	"Z":          2,
	"VISIBILITY": 3,
	"PRESENCE":   4,
}

func (x LandmarksToTensorCalculatorOptions_Attribute) Enum() *LandmarksToTensorCalculatorOptions_Attribute {
	p := new(LandmarksToTensorCalculatorOptions_Attribute)
	*p = x
	return p
}

func (x LandmarksToTensorCalculatorOptions_Attribute) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(LandmarksToTensorCalculatorOptions_Attribute_name, int32(x))
}

func (x *LandmarksToTensorCalculatorOptions_Attribute) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(LandmarksToTensorCalculatorOptions_Attribute_value, data, "LandmarksToTensorCalculatorOptions_Attribute")
	if err != nil {
		return err
	}
	*x = LandmarksToTensorCalculatorOptions_Attribute(value)
	return nil
}

func (LandmarksToTensorCalculatorOptions_Attribute) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_24fd018a03505cd8, []int{0, 0}
}

type LandmarksToTensorCalculatorOptions struct {
	Attributes []LandmarksToTensorCalculatorOptions_Attribute `protobuf:"varint,1,rep,name=attributes,enum=mediapipe.LandmarksToTensorCalculatorOptions_Attribute" json:"attributes,omitempty"`
	Flatten    *bool                                          `protobuf:"varint,2,opt,name=flatten,def=0" json:"flatten,omitempty"`
}

func (m *LandmarksToTensorCalculatorOptions) Reset()      { *m = LandmarksToTensorCalculatorOptions{} }
func (*LandmarksToTensorCalculatorOptions) ProtoMessage() {}
func (*LandmarksToTensorCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_24fd018a03505cd8, []int{0}
}
func (m *LandmarksToTensorCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *LandmarksToTensorCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_LandmarksToTensorCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *LandmarksToTensorCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LandmarksToTensorCalculatorOptions.Merge(m, src)
}
func (m *LandmarksToTensorCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *LandmarksToTensorCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_LandmarksToTensorCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_LandmarksToTensorCalculatorOptions proto.InternalMessageInfo

const Default_LandmarksToTensorCalculatorOptions_Flatten bool = false

func (m *LandmarksToTensorCalculatorOptions) GetAttributes() []LandmarksToTensorCalculatorOptions_Attribute {
	if m != nil {
		return m.Attributes
	}
	return nil
}

func (m *LandmarksToTensorCalculatorOptions) GetFlatten() bool {
	if m != nil && m.Flatten != nil {
		return *m.Flatten
	}
	return Default_LandmarksToTensorCalculatorOptions_Flatten
}

var E_LandmarksToTensorCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*LandmarksToTensorCalculatorOptions)(nil),
	Field:         394810235,
	Name:          "mediapipe.LandmarksToTensorCalculatorOptions.ext",
	Tag:           "bytes,394810235,opt,name=ext",
	Filename:      "mediapipe/calculators/tensor/landmarks_to_tensor_calculator.proto",
}

func init() {
	proto.RegisterEnum("mediapipe.LandmarksToTensorCalculatorOptions_Attribute", LandmarksToTensorCalculatorOptions_Attribute_name, LandmarksToTensorCalculatorOptions_Attribute_value)
	proto.RegisterExtension(E_LandmarksToTensorCalculatorOptions_Ext)
	proto.RegisterType((*LandmarksToTensorCalculatorOptions)(nil), "mediapipe.LandmarksToTensorCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/tensor/landmarks_to_tensor_calculator.proto", fileDescriptor_24fd018a03505cd8)
}

var fileDescriptor_24fd018a03505cd8 = []byte{
	// 352 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x72, 0xcc, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0x2f, 0x49, 0xcd, 0x2b, 0xce, 0x2f, 0xd2, 0xcf, 0x49, 0xcc, 0x4b, 0xc9, 0x4d, 0x2c,
	0xca, 0x2e, 0x8e, 0x2f, 0xc9, 0x8f, 0x87, 0x88, 0xc5, 0x23, 0x94, 0xe9, 0x15, 0x14, 0xe5, 0x97,
	0xe4, 0x0b, 0x71, 0xc2, 0x8d, 0x90, 0x52, 0x41, 0x98, 0x96, 0x56, 0x94, 0x98, 0x9b, 0x5a, 0x9e,
	0x5f, 0x94, 0xad, 0x8f, 0xae, 0x41, 0xe9, 0x20, 0x13, 0x97, 0x92, 0x0f, 0xcc, 0xe4, 0x90, 0xfc,
	0x10, 0xb0, 0xb9, 0xce, 0x70, 0x55, 0xfe, 0x05, 0x25, 0x99, 0xf9, 0x79, 0xc5, 0x42, 0xe1, 0x5c,
	0x5c, 0x89, 0x25, 0x25, 0x45, 0x99, 0x49, 0xa5, 0x25, 0xa9, 0xc5, 0x12, 0x8c, 0x0a, 0xcc, 0x1a,
	0x7c, 0x46, 0xe6, 0x7a, 0x70, 0x1b, 0xf4, 0x08, 0x1b, 0xa1, 0xe7, 0x08, 0xd3, 0x1f, 0x84, 0x64,
	0x94, 0x90, 0x3c, 0x17, 0x7b, 0x5a, 0x4e, 0x62, 0x49, 0x49, 0x6a, 0x9e, 0x04, 0x93, 0x02, 0xa3,
	0x06, 0x87, 0x15, 0x6b, 0x5a, 0x62, 0x4e, 0x71, 0x6a, 0x10, 0x4c, 0x54, 0xc9, 0x8e, 0x8b, 0x13,
	0xae, 0x53, 0x88, 0x95, 0x8b, 0x31, 0x42, 0x80, 0x01, 0x44, 0x45, 0x0a, 0x30, 0x82, 0xa8, 0x28,
	0x01, 0x26, 0x21, 0x3e, 0x2e, 0xae, 0x30, 0xcf, 0x60, 0x4f, 0x27, 0x4f, 0x1f, 0xcf, 0x90, 0x48,
	0x01, 0x66, 0x21, 0x1e, 0x2e, 0x8e, 0x80, 0x20, 0xd7, 0x60, 0x57, 0x3f, 0x67, 0x57, 0x01, 0x16,
	0xa3, 0x44, 0x2e, 0xe6, 0xd4, 0x8a, 0x12, 0x21, 0x19, 0x24, 0xc7, 0x62, 0xb8, 0x4d, 0xe2, 0xf7,
	0xb2, 0x85, 0x7b, 0x18, 0x15, 0x18, 0x35, 0xb8, 0x8d, 0x74, 0x49, 0xf2, 0x54, 0x10, 0xc8, 0x6c,
	0xa7, 0xbc, 0x0b, 0x0f, 0xe5, 0x18, 0x6e, 0x3c, 0x94, 0x63, 0xf8, 0xf0, 0x50, 0x8e, 0xb1, 0xe1,
	0x91, 0x1c, 0xe3, 0x8a, 0x47, 0x72, 0x8c, 0x27, 0x1e, 0xc9, 0x31, 0x5e, 0x78, 0x24, 0xc7, 0xf8,
	0xe0, 0x91, 0x1c, 0xe3, 0x8b, 0x47, 0x72, 0x0c, 0x1f, 0x1e, 0xc9, 0x31, 0x4e, 0x78, 0x2c, 0xc7,
	0x70, 0xe1, 0xb1, 0x1c, 0xc3, 0x8d, 0xc7, 0x72, 0x0c, 0x51, 0x16, 0xe9, 0x99, 0x25, 0x19, 0xa5,
	0x49, 0x7a, 0xc9, 0xf9, 0xb9, 0xfa, 0xe9, 0xf9, 0xf9, 0xe9, 0x39, 0xa9, 0xfa, 0x88, 0x18, 0xc3,
	0x97, 0x12, 0x00, 0x01, 0x00, 0x00, 0xff, 0xff, 0x6b, 0x50, 0x6f, 0xcc, 0x28, 0x02, 0x00, 0x00,
}

func (x LandmarksToTensorCalculatorOptions_Attribute) String() string {
	s, ok := LandmarksToTensorCalculatorOptions_Attribute_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *LandmarksToTensorCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*LandmarksToTensorCalculatorOptions)
	if !ok {
		that2, ok := that.(LandmarksToTensorCalculatorOptions)
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
	if len(this.Attributes) != len(that1.Attributes) {
		return false
	}
	for i := range this.Attributes {
		if this.Attributes[i] != that1.Attributes[i] {
			return false
		}
	}
	if this.Flatten != nil && that1.Flatten != nil {
		if *this.Flatten != *that1.Flatten {
			return false
		}
	} else if this.Flatten != nil {
		return false
	} else if that1.Flatten != nil {
		return false
	}
	return true
}
func (this *LandmarksToTensorCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&tensor.LandmarksToTensorCalculatorOptions{")
	if this.Attributes != nil {
		s = append(s, "Attributes: "+fmt.Sprintf("%#v", this.Attributes)+",\n")
	}
	if this.Flatten != nil {
		s = append(s, "Flatten: "+valueToGoStringLandmarksToTensorCalculator(this.Flatten, "bool")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringLandmarksToTensorCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *LandmarksToTensorCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *LandmarksToTensorCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *LandmarksToTensorCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Flatten != nil {
		i--
		if *m.Flatten {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x10
	}
	if len(m.Attributes) > 0 {
		for iNdEx := len(m.Attributes) - 1; iNdEx >= 0; iNdEx-- {
			i = encodeVarintLandmarksToTensorCalculator(dAtA, i, uint64(m.Attributes[iNdEx]))
			i--
			dAtA[i] = 0x8
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintLandmarksToTensorCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovLandmarksToTensorCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *LandmarksToTensorCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Attributes) > 0 {
		for _, e := range m.Attributes {
			n += 1 + sovLandmarksToTensorCalculator(uint64(e))
		}
	}
	if m.Flatten != nil {
		n += 2
	}
	return n
}

func sovLandmarksToTensorCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozLandmarksToTensorCalculator(x uint64) (n int) {
	return sovLandmarksToTensorCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *LandmarksToTensorCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&LandmarksToTensorCalculatorOptions{`,
		`Attributes:` + fmt.Sprintf("%v", this.Attributes) + `,`,
		`Flatten:` + valueToStringLandmarksToTensorCalculator(this.Flatten) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringLandmarksToTensorCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *LandmarksToTensorCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLandmarksToTensorCalculator
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
			return fmt.Errorf("proto: LandmarksToTensorCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: LandmarksToTensorCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType == 0 {
				var v LandmarksToTensorCalculatorOptions_Attribute
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowLandmarksToTensorCalculator
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					v |= LandmarksToTensorCalculatorOptions_Attribute(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.Attributes = append(m.Attributes, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowLandmarksToTensorCalculator
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
					return ErrInvalidLengthLandmarksToTensorCalculator
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthLandmarksToTensorCalculator
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				if elementCount != 0 && len(m.Attributes) == 0 {
					m.Attributes = make([]LandmarksToTensorCalculatorOptions_Attribute, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v LandmarksToTensorCalculatorOptions_Attribute
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowLandmarksToTensorCalculator
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						v |= LandmarksToTensorCalculatorOptions_Attribute(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.Attributes = append(m.Attributes, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field Attributes", wireType)
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Flatten", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLandmarksToTensorCalculator
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
			m.Flatten = &b
		default:
			iNdEx = preIndex
			skippy, err := skipLandmarksToTensorCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthLandmarksToTensorCalculator
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
func skipLandmarksToTensorCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowLandmarksToTensorCalculator
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
					return 0, ErrIntOverflowLandmarksToTensorCalculator
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
					return 0, ErrIntOverflowLandmarksToTensorCalculator
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
				return 0, ErrInvalidLengthLandmarksToTensorCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupLandmarksToTensorCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthLandmarksToTensorCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthLandmarksToTensorCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowLandmarksToTensorCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupLandmarksToTensorCalculator = fmt.Errorf("proto: unexpected end of group")
)