// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/internal/callback_packet_calculator.proto

package internal

import (
	bytes "bytes"
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

type CallbackPacketCalculatorOptions_PointerType int32

const (
	UNKNOWN            CallbackPacketCalculatorOptions_PointerType = 0
	VECTOR_PACKET      CallbackPacketCalculatorOptions_PointerType = 1
	POST_STREAM_PACKET CallbackPacketCalculatorOptions_PointerType = 2
)

var CallbackPacketCalculatorOptions_PointerType_name = map[int32]string{
	0: "UNKNOWN",
	1: "VECTOR_PACKET",
	2: "POST_STREAM_PACKET",
}

var CallbackPacketCalculatorOptions_PointerType_value = map[string]int32{
	"UNKNOWN":            0,
	"VECTOR_PACKET":      1,
	"POST_STREAM_PACKET": 2,
}

func (x CallbackPacketCalculatorOptions_PointerType) Enum() *CallbackPacketCalculatorOptions_PointerType {
	p := new(CallbackPacketCalculatorOptions_PointerType)
	*p = x
	return p
}

func (x CallbackPacketCalculatorOptions_PointerType) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(CallbackPacketCalculatorOptions_PointerType_name, int32(x))
}

func (x *CallbackPacketCalculatorOptions_PointerType) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(CallbackPacketCalculatorOptions_PointerType_value, data, "CallbackPacketCalculatorOptions_PointerType")
	if err != nil {
		return err
	}
	*x = CallbackPacketCalculatorOptions_PointerType(value)
	return nil
}

func (CallbackPacketCalculatorOptions_PointerType) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_3d34a6dba4f36cb9, []int{0, 0}
}

type CallbackPacketCalculatorOptions struct {
	Type    CallbackPacketCalculatorOptions_PointerType `protobuf:"varint,1,opt,name=type,enum=mediapipe.CallbackPacketCalculatorOptions_PointerType" json:"type"`
	Pointer []byte                                      `protobuf:"bytes,2,opt,name=pointer" json:"pointer"`
}

func (m *CallbackPacketCalculatorOptions) Reset()      { *m = CallbackPacketCalculatorOptions{} }
func (*CallbackPacketCalculatorOptions) ProtoMessage() {}
func (*CallbackPacketCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_3d34a6dba4f36cb9, []int{0}
}
func (m *CallbackPacketCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *CallbackPacketCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_CallbackPacketCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *CallbackPacketCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CallbackPacketCalculatorOptions.Merge(m, src)
}
func (m *CallbackPacketCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *CallbackPacketCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_CallbackPacketCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_CallbackPacketCalculatorOptions proto.InternalMessageInfo

func (m *CallbackPacketCalculatorOptions) GetType() CallbackPacketCalculatorOptions_PointerType {
	if m != nil {
		return m.Type
	}
	return UNKNOWN
}

func (m *CallbackPacketCalculatorOptions) GetPointer() []byte {
	if m != nil {
		return m.Pointer
	}
	return nil
}

var E_CallbackPacketCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*CallbackPacketCalculatorOptions)(nil),
	Field:         245965803,
	Name:          "mediapipe.CallbackPacketCalculatorOptions.ext",
	Tag:           "bytes,245965803,opt,name=ext",
	Filename:      "mediapipe/calculators/internal/callback_packet_calculator.proto",
}

func init() {
	proto.RegisterEnum("mediapipe.CallbackPacketCalculatorOptions_PointerType", CallbackPacketCalculatorOptions_PointerType_name, CallbackPacketCalculatorOptions_PointerType_value)
	proto.RegisterExtension(E_CallbackPacketCalculatorOptions_Ext)
	proto.RegisterType((*CallbackPacketCalculatorOptions)(nil), "mediapipe.CallbackPacketCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/internal/callback_packet_calculator.proto", fileDescriptor_3d34a6dba4f36cb9)
}

var fileDescriptor_3d34a6dba4f36cb9 = []byte{
	// 347 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xb2, 0xcf, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0xcf, 0xcc, 0x2b, 0x49, 0x2d, 0xca, 0x4b, 0xcc, 0x01, 0x09, 0xe6, 0x24, 0x25, 0x26,
	0x67, 0xc7, 0x17, 0x24, 0x26, 0x67, 0xa7, 0x96, 0xc4, 0x23, 0x14, 0xe9, 0x15, 0x14, 0xe5, 0x97,
	0xe4, 0x0b, 0x71, 0xc2, 0x0d, 0x90, 0x52, 0x41, 0x98, 0x95, 0x56, 0x94, 0x98, 0x9b, 0x5a, 0x9e,
	0x5f, 0x94, 0xad, 0x8f, 0xae, 0x41, 0x69, 0x27, 0x13, 0x97, 0xbc, 0x33, 0xd4, 0xd4, 0x00, 0xb0,
	0xa1, 0xce, 0x70, 0x25, 0xfe, 0x05, 0x25, 0x99, 0xf9, 0x79, 0xc5, 0x42, 0x01, 0x5c, 0x2c, 0x25,
	0x95, 0x05, 0xa9, 0x12, 0x8c, 0x0a, 0x8c, 0x1a, 0x7c, 0x46, 0x66, 0x7a, 0x70, 0x83, 0xf5, 0x08,
	0xe8, 0xd4, 0x0b, 0xc8, 0x07, 0x3b, 0x3d, 0xa4, 0xb2, 0x20, 0xd5, 0x89, 0xe5, 0xc4, 0x3d, 0x79,
	0x86, 0x20, 0xb0, 0x49, 0x42, 0x72, 0x5c, 0xec, 0x05, 0x10, 0x29, 0x09, 0x26, 0x05, 0x46, 0x0d,
	0x1e, 0xa8, 0x24, 0x4c, 0x50, 0xc9, 0x95, 0x8b, 0x1b, 0x49, 0xab, 0x10, 0x37, 0x17, 0x7b, 0xa8,
	0x9f, 0xb7, 0x9f, 0x7f, 0xb8, 0x9f, 0x00, 0x83, 0x90, 0x20, 0x17, 0x6f, 0x98, 0xab, 0x73, 0x88,
	0x7f, 0x50, 0x7c, 0x80, 0xa3, 0xb3, 0xb7, 0x6b, 0x88, 0x00, 0xa3, 0x90, 0x18, 0x97, 0x50, 0x80,
	0x7f, 0x70, 0x48, 0x7c, 0x70, 0x48, 0x90, 0xab, 0xa3, 0x2f, 0x4c, 0x9c, 0xc9, 0x28, 0x96, 0x8b,
	0x39, 0xb5, 0xa2, 0x44, 0x48, 0x06, 0xd5, 0xc5, 0xa8, 0x6e, 0x94, 0x78, 0x7d, 0x7c, 0x49, 0xa9,
	0x02, 0xa3, 0x06, 0xb7, 0x91, 0x16, 0xf1, 0x1e, 0x0b, 0x02, 0x99, 0xeb, 0x54, 0x70, 0xe1, 0xa1,
	0x1c, 0xc3, 0x8d, 0x87, 0x72, 0x0c, 0x1f, 0x1e, 0xca, 0x31, 0x36, 0x3c, 0x92, 0x63, 0x5c, 0xf1,
	0x48, 0x8e, 0xf1, 0xc4, 0x23, 0x39, 0xc6, 0x0b, 0x8f, 0xe4, 0x18, 0x1f, 0x3c, 0x92, 0x63, 0x7c,
	0xf1, 0x48, 0x8e, 0xe1, 0xc3, 0x23, 0x39, 0xc6, 0x09, 0x8f, 0xe5, 0x18, 0x2e, 0x3c, 0x96, 0x63,
	0xb8, 0xf1, 0x58, 0x8e, 0x21, 0xca, 0x2a, 0x3d, 0xb3, 0x24, 0xa3, 0x34, 0x49, 0x2f, 0x39, 0x3f,
	0x57, 0x3f, 0x3d, 0x3f, 0x3f, 0x3d, 0x27, 0x55, 0x1f, 0x11, 0x53, 0xf8, 0xe3, 0x1f, 0x10, 0x00,
	0x00, 0xff, 0xff, 0xb8, 0xc7, 0xab, 0xcf, 0x20, 0x02, 0x00, 0x00,
}

func (x CallbackPacketCalculatorOptions_PointerType) String() string {
	s, ok := CallbackPacketCalculatorOptions_PointerType_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *CallbackPacketCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*CallbackPacketCalculatorOptions)
	if !ok {
		that2, ok := that.(CallbackPacketCalculatorOptions)
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
	if this.Type != that1.Type {
		return false
	}
	if !bytes.Equal(this.Pointer, that1.Pointer) {
		return false
	}
	return true
}
func (this *CallbackPacketCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&internal.CallbackPacketCalculatorOptions{")
	s = append(s, "Type: "+fmt.Sprintf("%#v", this.Type)+",\n")
	s = append(s, "Pointer: "+fmt.Sprintf("%#v", this.Pointer)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringCallbackPacketCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *CallbackPacketCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *CallbackPacketCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *CallbackPacketCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Pointer != nil {
		i -= len(m.Pointer)
		copy(dAtA[i:], m.Pointer)
		i = encodeVarintCallbackPacketCalculator(dAtA, i, uint64(len(m.Pointer)))
		i--
		dAtA[i] = 0x12
	}
	i = encodeVarintCallbackPacketCalculator(dAtA, i, uint64(m.Type))
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func encodeVarintCallbackPacketCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovCallbackPacketCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *CallbackPacketCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 1 + sovCallbackPacketCalculator(uint64(m.Type))
	if m.Pointer != nil {
		l = len(m.Pointer)
		n += 1 + l + sovCallbackPacketCalculator(uint64(l))
	}
	return n
}

func sovCallbackPacketCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozCallbackPacketCalculator(x uint64) (n int) {
	return sovCallbackPacketCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *CallbackPacketCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&CallbackPacketCalculatorOptions{`,
		`Type:` + fmt.Sprintf("%v", this.Type) + `,`,
		`Pointer:` + fmt.Sprintf("%v", this.Pointer) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringCallbackPacketCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *CallbackPacketCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCallbackPacketCalculator
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
			return fmt.Errorf("proto: CallbackPacketCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: CallbackPacketCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Type", wireType)
			}
			m.Type = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCallbackPacketCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Type |= CallbackPacketCalculatorOptions_PointerType(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Pointer", wireType)
			}
			var byteLen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCallbackPacketCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				byteLen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if byteLen < 0 {
				return ErrInvalidLengthCallbackPacketCalculator
			}
			postIndex := iNdEx + byteLen
			if postIndex < 0 {
				return ErrInvalidLengthCallbackPacketCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Pointer = append(m.Pointer[:0], dAtA[iNdEx:postIndex]...)
			if m.Pointer == nil {
				m.Pointer = []byte{}
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipCallbackPacketCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthCallbackPacketCalculator
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
func skipCallbackPacketCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowCallbackPacketCalculator
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
					return 0, ErrIntOverflowCallbackPacketCalculator
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
					return 0, ErrIntOverflowCallbackPacketCalculator
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
				return 0, ErrInvalidLengthCallbackPacketCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupCallbackPacketCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthCallbackPacketCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthCallbackPacketCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowCallbackPacketCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupCallbackPacketCalculator = fmt.Errorf("proto: unexpected end of group")
)