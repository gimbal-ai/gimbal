// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/core/gate_calculator.proto

package core

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

type GateCalculatorOptions_GateState int32

const (
	UNSPECIFIED        GateCalculatorOptions_GateState = 0
	GATE_UNINITIALIZED GateCalculatorOptions_GateState = 1
	GATE_ALLOW         GateCalculatorOptions_GateState = 2
	GATE_DISALLOW      GateCalculatorOptions_GateState = 3
)

var GateCalculatorOptions_GateState_name = map[int32]string{
	0: "UNSPECIFIED",
	1: "GATE_UNINITIALIZED",
	2: "GATE_ALLOW",
	3: "GATE_DISALLOW",
}

var GateCalculatorOptions_GateState_value = map[string]int32{
	"UNSPECIFIED":        0,
	"GATE_UNINITIALIZED": 1,
	"GATE_ALLOW":         2,
	"GATE_DISALLOW":      3,
}

func (x GateCalculatorOptions_GateState) Enum() *GateCalculatorOptions_GateState {
	p := new(GateCalculatorOptions_GateState)
	*p = x
	return p
}

func (x GateCalculatorOptions_GateState) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(GateCalculatorOptions_GateState_name, int32(x))
}

func (x *GateCalculatorOptions_GateState) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(GateCalculatorOptions_GateState_value, data, "GateCalculatorOptions_GateState")
	if err != nil {
		return err
	}
	*x = GateCalculatorOptions_GateState(value)
	return nil
}

func (GateCalculatorOptions_GateState) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_659c2055944969a9, []int{0, 0}
}

type GateCalculatorOptions struct {
	EmptyPacketsAsAllow bool                             `protobuf:"varint,1,opt,name=empty_packets_as_allow,json=emptyPacketsAsAllow" json:"empty_packets_as_allow"`
	Allow               *bool                            `protobuf:"varint,2,opt,name=allow,def=0" json:"allow,omitempty"`
	InitialGateState    *GateCalculatorOptions_GateState `protobuf:"varint,3,opt,name=initial_gate_state,json=initialGateState,enum=mediapipe.GateCalculatorOptions_GateState,def=1" json:"initial_gate_state,omitempty"`
}

func (m *GateCalculatorOptions) Reset()      { *m = GateCalculatorOptions{} }
func (*GateCalculatorOptions) ProtoMessage() {}
func (*GateCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_659c2055944969a9, []int{0}
}
func (m *GateCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *GateCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_GateCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *GateCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GateCalculatorOptions.Merge(m, src)
}
func (m *GateCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *GateCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_GateCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_GateCalculatorOptions proto.InternalMessageInfo

const Default_GateCalculatorOptions_Allow bool = false
const Default_GateCalculatorOptions_InitialGateState GateCalculatorOptions_GateState = GATE_UNINITIALIZED

func (m *GateCalculatorOptions) GetEmptyPacketsAsAllow() bool {
	if m != nil {
		return m.EmptyPacketsAsAllow
	}
	return false
}

func (m *GateCalculatorOptions) GetAllow() bool {
	if m != nil && m.Allow != nil {
		return *m.Allow
	}
	return Default_GateCalculatorOptions_Allow
}

func (m *GateCalculatorOptions) GetInitialGateState() GateCalculatorOptions_GateState {
	if m != nil && m.InitialGateState != nil {
		return *m.InitialGateState
	}
	return Default_GateCalculatorOptions_InitialGateState
}

var E_GateCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*GateCalculatorOptions)(nil),
	Field:         261754847,
	Name:          "mediapipe.GateCalculatorOptions.ext",
	Tag:           "bytes,261754847,opt,name=ext",
	Filename:      "mediapipe/calculators/core/gate_calculator.proto",
}

func init() {
	proto.RegisterEnum("mediapipe.GateCalculatorOptions_GateState", GateCalculatorOptions_GateState_name, GateCalculatorOptions_GateState_value)
	proto.RegisterExtension(E_GateCalculatorOptions_Ext)
	proto.RegisterType((*GateCalculatorOptions)(nil), "mediapipe.GateCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/core/gate_calculator.proto", fileDescriptor_659c2055944969a9)
}

var fileDescriptor_659c2055944969a9 = []byte{
	// 405 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x92, 0x41, 0x8b, 0xd3, 0x40,
	0x1c, 0xc5, 0x67, 0x36, 0x2e, 0xb8, 0xb3, 0xb8, 0xc6, 0x11, 0x97, 0xa2, 0x32, 0x86, 0xc5, 0x43,
	0xf1, 0x90, 0x48, 0x0f, 0x82, 0x7b, 0xcb, 0xb6, 0xb1, 0x04, 0x4a, 0x5b, 0x92, 0x96, 0x42, 0x2f,
	0x61, 0x8c, 0xd3, 0x18, 0x3a, 0xe9, 0x84, 0x64, 0x4a, 0x15, 0x3c, 0x78, 0xf7, 0xe2, 0x37, 0xf0,
	0xea, 0x47, 0xe9, 0xb1, 0xc7, 0x9e, 0xd4, 0xa6, 0x97, 0x1e, 0xfb, 0x11, 0x24, 0x89, 0xa4, 0xa2,
	0x15, 0x8f, 0xff, 0x37, 0xbf, 0xf7, 0x78, 0x7f, 0xe6, 0x8f, 0x9e, 0x47, 0xec, 0x4d, 0x48, 0xe3,
	0x30, 0x66, 0x86, 0x4f, 0xb9, 0x3f, 0xe7, 0x54, 0x8a, 0x24, 0x35, 0x7c, 0x91, 0x30, 0x23, 0xa0,
	0x92, 0x79, 0x07, 0x55, 0x8f, 0x13, 0x21, 0x05, 0x3e, 0xab, 0x1c, 0x0f, 0x9f, 0x1e, 0xcc, 0x93,
	0x84, 0x46, 0x6c, 0x21, 0x92, 0xa9, 0xf1, 0xa7, 0xe1, 0xea, 0x93, 0x82, 0x1e, 0xb4, 0xa9, 0x64,
	0xcd, 0xea, 0xa1, 0x17, 0xcb, 0x50, 0xcc, 0x52, 0xfc, 0x12, 0x5d, 0xb2, 0x28, 0x96, 0xef, 0xbd,
	0x98, 0xfa, 0x53, 0x26, 0x53, 0x8f, 0xa6, 0x1e, 0xe5, 0x5c, 0x2c, 0x6a, 0x50, 0x83, 0xf5, 0xdb,
	0x37, 0xb7, 0x96, 0xdf, 0x9e, 0x00, 0xe7, 0x7e, 0xc1, 0xf4, 0x4b, 0xc4, 0x4c, 0xcd, 0x1c, 0xc0,
	0x8f, 0xd0, 0x69, 0x49, 0x9e, 0xe4, 0xe4, 0xf5, 0xe9, 0x84, 0xf2, 0x94, 0x39, 0xa5, 0x86, 0x39,
	0xc2, 0xe1, 0x2c, 0x94, 0x21, 0xe5, 0x5e, 0xb1, 0x43, 0x2a, 0xa9, 0x64, 0x35, 0x45, 0x83, 0xf5,
	0x8b, 0xc6, 0x33, 0xbd, 0x2a, 0xad, 0x1f, 0x6d, 0x55, 0xa8, 0x6e, 0xee, 0xb8, 0xc6, 0x6d, 0x73,
	0x60, 0x79, 0xc3, 0xae, 0xdd, 0xb5, 0x07, 0xb6, 0xd9, 0xb1, 0xc7, 0x56, 0xcb, 0x51, 0x7f, 0x25,
	0x57, 0xd4, 0xd5, 0x08, 0x9d, 0x55, 0x03, 0xbe, 0x8b, 0xce, 0x87, 0x5d, 0xb7, 0x6f, 0x35, 0xed,
	0x57, 0xb6, 0xd5, 0x52, 0x01, 0xbe, 0x44, 0x47, 0x52, 0x54, 0x88, 0x2f, 0x10, 0x2a, 0x74, 0xb3,
	0xd3, 0xe9, 0x8d, 0xd4, 0x13, 0x7c, 0x0f, 0xdd, 0x29, 0xe6, 0x96, 0xed, 0x96, 0x92, 0xd2, 0x70,
	0x91, 0xc2, 0xde, 0x49, 0xfc, 0xf8, 0xb7, 0xc6, 0x7f, 0xb5, 0xad, 0x7d, 0xff, 0xb2, 0xfb, 0xa0,
	0xc1, 0xfa, 0x79, 0x43, 0xfb, 0xdf, 0x62, 0x4e, 0x9e, 0x76, 0xc3, 0x57, 0x1b, 0x02, 0xd6, 0x1b,
	0x02, 0xf6, 0x1b, 0x02, 0x3f, 0x66, 0x04, 0x7e, 0xcd, 0x08, 0x5c, 0x66, 0x04, 0xae, 0x32, 0x02,
	0x7f, 0x64, 0x04, 0xee, 0x32, 0x02, 0xf6, 0x19, 0x81, 0x9f, 0xb7, 0x04, 0xac, 0xb6, 0x04, 0xac,
	0xb7, 0x04, 0x8c, 0x5f, 0x04, 0xa1, 0x7c, 0x3b, 0x7f, 0xad, 0xfb, 0x22, 0x32, 0x02, 0x21, 0x02,
	0xce, 0x8c, 0xc3, 0xdf, 0xff, 0xfb, 0x84, 0x7e, 0x06, 0x00, 0x00, 0xff, 0xff, 0xed, 0x29, 0x90,
	0xfa, 0x5f, 0x02, 0x00, 0x00,
}

func (x GateCalculatorOptions_GateState) String() string {
	s, ok := GateCalculatorOptions_GateState_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *GateCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*GateCalculatorOptions)
	if !ok {
		that2, ok := that.(GateCalculatorOptions)
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
	if this.EmptyPacketsAsAllow != that1.EmptyPacketsAsAllow {
		return false
	}
	if this.Allow != nil && that1.Allow != nil {
		if *this.Allow != *that1.Allow {
			return false
		}
	} else if this.Allow != nil {
		return false
	} else if that1.Allow != nil {
		return false
	}
	if this.InitialGateState != nil && that1.InitialGateState != nil {
		if *this.InitialGateState != *that1.InitialGateState {
			return false
		}
	} else if this.InitialGateState != nil {
		return false
	} else if that1.InitialGateState != nil {
		return false
	}
	return true
}
func (this *GateCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&core.GateCalculatorOptions{")
	s = append(s, "EmptyPacketsAsAllow: "+fmt.Sprintf("%#v", this.EmptyPacketsAsAllow)+",\n")
	if this.Allow != nil {
		s = append(s, "Allow: "+valueToGoStringGateCalculator(this.Allow, "bool")+",\n")
	}
	if this.InitialGateState != nil {
		s = append(s, "InitialGateState: "+valueToGoStringGateCalculator(this.InitialGateState, "GateCalculatorOptions_GateState")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringGateCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *GateCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *GateCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *GateCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.InitialGateState != nil {
		i = encodeVarintGateCalculator(dAtA, i, uint64(*m.InitialGateState))
		i--
		dAtA[i] = 0x18
	}
	if m.Allow != nil {
		i--
		if *m.Allow {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x10
	}
	i--
	if m.EmptyPacketsAsAllow {
		dAtA[i] = 1
	} else {
		dAtA[i] = 0
	}
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func encodeVarintGateCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovGateCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *GateCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 2
	if m.Allow != nil {
		n += 2
	}
	if m.InitialGateState != nil {
		n += 1 + sovGateCalculator(uint64(*m.InitialGateState))
	}
	return n
}

func sovGateCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozGateCalculator(x uint64) (n int) {
	return sovGateCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *GateCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&GateCalculatorOptions{`,
		`EmptyPacketsAsAllow:` + fmt.Sprintf("%v", this.EmptyPacketsAsAllow) + `,`,
		`Allow:` + valueToStringGateCalculator(this.Allow) + `,`,
		`InitialGateState:` + valueToStringGateCalculator(this.InitialGateState) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringGateCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *GateCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowGateCalculator
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
			return fmt.Errorf("proto: GateCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: GateCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field EmptyPacketsAsAllow", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowGateCalculator
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
			m.EmptyPacketsAsAllow = bool(v != 0)
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Allow", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowGateCalculator
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
			m.Allow = &b
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field InitialGateState", wireType)
			}
			var v GateCalculatorOptions_GateState
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowGateCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= GateCalculatorOptions_GateState(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.InitialGateState = &v
		default:
			iNdEx = preIndex
			skippy, err := skipGateCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthGateCalculator
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
func skipGateCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowGateCalculator
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
					return 0, ErrIntOverflowGateCalculator
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
					return 0, ErrIntOverflowGateCalculator
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
				return 0, ErrInvalidLengthGateCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupGateCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthGateCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthGateCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowGateCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupGateCalculator = fmt.Errorf("proto: unexpected end of group")
)
