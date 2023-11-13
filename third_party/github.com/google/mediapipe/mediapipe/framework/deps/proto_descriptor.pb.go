// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/deps/proto_descriptor.proto

package deps

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
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

type FieldDescriptorProto_Type int32

const (
	TYPE_INVALID  FieldDescriptorProto_Type = 0
	TYPE_DOUBLE   FieldDescriptorProto_Type = 1
	TYPE_FLOAT    FieldDescriptorProto_Type = 2
	TYPE_INT64    FieldDescriptorProto_Type = 3
	TYPE_UINT64   FieldDescriptorProto_Type = 4
	TYPE_INT32    FieldDescriptorProto_Type = 5
	TYPE_FIXED64  FieldDescriptorProto_Type = 6
	TYPE_FIXED32  FieldDescriptorProto_Type = 7
	TYPE_BOOL     FieldDescriptorProto_Type = 8
	TYPE_STRING   FieldDescriptorProto_Type = 9
	TYPE_GROUP    FieldDescriptorProto_Type = 10
	TYPE_MESSAGE  FieldDescriptorProto_Type = 11
	TYPE_BYTES    FieldDescriptorProto_Type = 12
	TYPE_UINT32   FieldDescriptorProto_Type = 13
	TYPE_ENUM     FieldDescriptorProto_Type = 14
	TYPE_SFIXED32 FieldDescriptorProto_Type = 15
	TYPE_SFIXED64 FieldDescriptorProto_Type = 16
	TYPE_SINT32   FieldDescriptorProto_Type = 17
	TYPE_SINT64   FieldDescriptorProto_Type = 18
)

var FieldDescriptorProto_Type_name = map[int32]string{
	0:  "TYPE_INVALID",
	1:  "TYPE_DOUBLE",
	2:  "TYPE_FLOAT",
	3:  "TYPE_INT64",
	4:  "TYPE_UINT64",
	5:  "TYPE_INT32",
	6:  "TYPE_FIXED64",
	7:  "TYPE_FIXED32",
	8:  "TYPE_BOOL",
	9:  "TYPE_STRING",
	10: "TYPE_GROUP",
	11: "TYPE_MESSAGE",
	12: "TYPE_BYTES",
	13: "TYPE_UINT32",
	14: "TYPE_ENUM",
	15: "TYPE_SFIXED32",
	16: "TYPE_SFIXED64",
	17: "TYPE_SINT32",
	18: "TYPE_SINT64",
}

var FieldDescriptorProto_Type_value = map[string]int32{
	"TYPE_INVALID":  0,
	"TYPE_DOUBLE":   1,
	"TYPE_FLOAT":    2,
	"TYPE_INT64":    3,
	"TYPE_UINT64":   4,
	"TYPE_INT32":    5,
	"TYPE_FIXED64":  6,
	"TYPE_FIXED32":  7,
	"TYPE_BOOL":     8,
	"TYPE_STRING":   9,
	"TYPE_GROUP":    10,
	"TYPE_MESSAGE":  11,
	"TYPE_BYTES":    12,
	"TYPE_UINT32":   13,
	"TYPE_ENUM":     14,
	"TYPE_SFIXED32": 15,
	"TYPE_SFIXED64": 16,
	"TYPE_SINT32":   17,
	"TYPE_SINT64":   18,
}

func (x FieldDescriptorProto_Type) Enum() *FieldDescriptorProto_Type {
	p := new(FieldDescriptorProto_Type)
	*p = x
	return p
}

func (x FieldDescriptorProto_Type) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(FieldDescriptorProto_Type_name, int32(x))
}

func (x *FieldDescriptorProto_Type) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(FieldDescriptorProto_Type_value, data, "FieldDescriptorProto_Type")
	if err != nil {
		return err
	}
	*x = FieldDescriptorProto_Type(value)
	return nil
}

func (FieldDescriptorProto_Type) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_06483c115557b8cf, []int{0, 0}
}

type FieldDescriptorProto struct {
}

func (m *FieldDescriptorProto) Reset()      { *m = FieldDescriptorProto{} }
func (*FieldDescriptorProto) ProtoMessage() {}
func (*FieldDescriptorProto) Descriptor() ([]byte, []int) {
	return fileDescriptor_06483c115557b8cf, []int{0}
}
func (m *FieldDescriptorProto) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *FieldDescriptorProto) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_FieldDescriptorProto.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *FieldDescriptorProto) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FieldDescriptorProto.Merge(m, src)
}
func (m *FieldDescriptorProto) XXX_Size() int {
	return m.Size()
}
func (m *FieldDescriptorProto) XXX_DiscardUnknown() {
	xxx_messageInfo_FieldDescriptorProto.DiscardUnknown(m)
}

var xxx_messageInfo_FieldDescriptorProto proto.InternalMessageInfo

func init() {
	proto.RegisterEnum("mediapipe.FieldDescriptorProto_Type", FieldDescriptorProto_Type_name, FieldDescriptorProto_Type_value)
	proto.RegisterType((*FieldDescriptorProto)(nil), "mediapipe.FieldDescriptorProto")
}

func init() {
	proto.RegisterFile("mediapipe/framework/deps/proto_descriptor.proto", fileDescriptor_06483c115557b8cf)
}

var fileDescriptor_06483c115557b8cf = []byte{
	// 372 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x92, 0xbd, 0x6e, 0xe2, 0x40,
	0x14, 0x85, 0x3d, 0x2c, 0xfb, 0xc3, 0xf0, 0x77, 0xb1, 0xb6, 0xd9, 0x2d, 0xa6, 0xe0, 0x01, 0x6c,
	0x09, 0x2c, 0xf7, 0x58, 0x36, 0xc8, 0x92, 0xb1, 0x11, 0xb6, 0x57, 0x4b, 0x1a, 0x44, 0xf0, 0x84,
	0x58, 0xc1, 0x1a, 0xcb, 0x10, 0x45, 0xe9, 0xf2, 0x08, 0x79, 0x85, 0x74, 0x79, 0x14, 0x4a, 0x4a,
	0xca, 0xd8, 0x34, 0x29, 0x79, 0x84, 0x28, 0xb6, 0x32, 0x81, 0x28, 0x69, 0x46, 0x9a, 0x4f, 0xe7,
	0x9e, 0x7b, 0xa4, 0x73, 0xb1, 0x1c, 0xd1, 0x20, 0x9c, 0xc5, 0x61, 0x4c, 0xe5, 0x8b, 0x64, 0x16,
	0xd1, 0x1b, 0x96, 0x5c, 0xc9, 0x01, 0x8d, 0x57, 0x72, 0x9c, 0xb0, 0x35, 0x9b, 0x06, 0x74, 0x35,
	0x4f, 0xc2, 0x78, 0xcd, 0x12, 0x29, 0x07, 0x62, 0x85, 0x0f, 0xb4, 0xd3, 0x12, 0xfe, 0xdd, 0x0f,
	0xe9, 0x32, 0xd0, 0xb9, 0x68, 0xf4, 0xaa, 0x69, 0x6f, 0x4a, 0xb8, 0xec, 0xdd, 0xc6, 0x54, 0x04,
	0x5c, 0xf3, 0x26, 0x23, 0x63, 0x6a, 0xda, 0xff, 0x7a, 0x96, 0xa9, 0x83, 0x20, 0x36, 0x71, 0x35,
	0x27, 0xba, 0xe3, 0x6b, 0x96, 0x01, 0x48, 0x6c, 0x60, 0x9c, 0x83, 0xbe, 0xe5, 0xf4, 0x3c, 0x28,
	0xf1, 0xbf, 0x69, 0x7b, 0xaa, 0x02, 0xdf, 0xf8, 0x80, 0x5f, 0x80, 0xf2, 0xb1, 0xa0, 0xdb, 0x81,
	0xef, 0x7c, 0x47, 0xdf, 0xfc, 0x6f, 0xe8, 0xaa, 0x02, 0x3f, 0x4e, 0x49, 0xb7, 0x03, 0x3f, 0xc5,
	0x3a, 0xae, 0xe4, 0x44, 0x73, 0x1c, 0x0b, 0x7e, 0x71, 0x4f, 0xd7, 0x1b, 0x9b, 0xf6, 0x00, 0x2a,
	0xdc, 0x73, 0x30, 0x76, 0xfc, 0x11, 0x60, 0xee, 0x30, 0x34, 0x5c, 0xb7, 0x37, 0x30, 0xa0, 0xca,
	0x15, 0xda, 0xc4, 0x33, 0x5c, 0xa8, 0x9d, 0xc4, 0xea, 0x76, 0xa0, 0xce, 0x57, 0x18, 0xb6, 0x3f,
	0x84, 0x86, 0xd8, 0xc2, 0xf5, 0x62, 0xc5, 0x5b, 0x88, 0xe6, 0x07, 0xa4, 0x2a, 0x00, 0xef, 0x41,
	0x0a, 0x97, 0xd6, 0x09, 0x50, 0x15, 0x10, 0xb5, 0x07, 0xb4, 0x4d, 0x89, 0xb0, 0x4b, 0x89, 0x70,
	0x48, 0x09, 0xba, 0xcb, 0x08, 0x7a, 0xcc, 0x08, 0xda, 0x64, 0x04, 0x6d, 0x33, 0x82, 0x9e, 0x32,
	0x82, 0x9e, 0x33, 0x22, 0x1c, 0x32, 0x82, 0xee, 0xf7, 0x44, 0xd8, 0xee, 0x89, 0xb0, 0xdb, 0x13,
	0x01, 0xff, 0x9d, 0xb3, 0x48, 0x5a, 0x30, 0xb6, 0x58, 0x52, 0x89, 0x17, 0x56, 0x34, 0xa8, 0xfd,
	0xf9, 0xac, 0xb3, 0xfc, 0x39, 0x53, 0x16, 0xe1, 0xfa, 0xf2, 0xfa, 0x5c, 0x9a, 0xb3, 0x48, 0x2e,
	0xa6, 0x8f, 0xee, 0xe3, 0xab, 0x4b, 0x79, 0x09, 0x00, 0x00, 0xff, 0xff, 0x2e, 0x67, 0xe0, 0x23,
	0x44, 0x02, 0x00, 0x00,
}

func (x FieldDescriptorProto_Type) String() string {
	s, ok := FieldDescriptorProto_Type_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *FieldDescriptorProto) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*FieldDescriptorProto)
	if !ok {
		that2, ok := that.(FieldDescriptorProto)
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
	return true
}
func (this *FieldDescriptorProto) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 4)
	s = append(s, "&deps.FieldDescriptorProto{")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringProtoDescriptor(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *FieldDescriptorProto) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *FieldDescriptorProto) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *FieldDescriptorProto) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	return len(dAtA) - i, nil
}

func encodeVarintProtoDescriptor(dAtA []byte, offset int, v uint64) int {
	offset -= sovProtoDescriptor(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *FieldDescriptorProto) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	return n
}

func sovProtoDescriptor(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozProtoDescriptor(x uint64) (n int) {
	return sovProtoDescriptor(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *FieldDescriptorProto) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&FieldDescriptorProto{`,
		`}`,
	}, "")
	return s
}
func valueToStringProtoDescriptor(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *FieldDescriptorProto) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowProtoDescriptor
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
			return fmt.Errorf("proto: FieldDescriptorProto: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: FieldDescriptorProto: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		default:
			iNdEx = preIndex
			skippy, err := skipProtoDescriptor(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthProtoDescriptor
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
func skipProtoDescriptor(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowProtoDescriptor
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
					return 0, ErrIntOverflowProtoDescriptor
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
					return 0, ErrIntOverflowProtoDescriptor
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
				return 0, ErrInvalidLengthProtoDescriptor
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupProtoDescriptor
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthProtoDescriptor
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthProtoDescriptor        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowProtoDescriptor          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupProtoDescriptor = fmt.Errorf("proto: unexpected end of group")
)
