// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/core/sequence_shift_calculator.proto

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

type SequenceShiftCalculatorOptions struct {
	PacketOffset                      *int32 `protobuf:"varint,1,opt,name=packet_offset,json=packetOffset,def=-1" json:"packet_offset,omitempty"`
	EmitEmptyPacketsBeforeFirstPacket *bool  `protobuf:"varint,2,opt,name=emit_empty_packets_before_first_packet,json=emitEmptyPacketsBeforeFirstPacket,def=0" json:"emit_empty_packets_before_first_packet,omitempty"`
}

func (m *SequenceShiftCalculatorOptions) Reset()      { *m = SequenceShiftCalculatorOptions{} }
func (*SequenceShiftCalculatorOptions) ProtoMessage() {}
func (*SequenceShiftCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_7e995a0b7788b71d, []int{0}
}
func (m *SequenceShiftCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SequenceShiftCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_SequenceShiftCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *SequenceShiftCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SequenceShiftCalculatorOptions.Merge(m, src)
}
func (m *SequenceShiftCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *SequenceShiftCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_SequenceShiftCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_SequenceShiftCalculatorOptions proto.InternalMessageInfo

const Default_SequenceShiftCalculatorOptions_PacketOffset int32 = -1
const Default_SequenceShiftCalculatorOptions_EmitEmptyPacketsBeforeFirstPacket bool = false

func (m *SequenceShiftCalculatorOptions) GetPacketOffset() int32 {
	if m != nil && m.PacketOffset != nil {
		return *m.PacketOffset
	}
	return Default_SequenceShiftCalculatorOptions_PacketOffset
}

func (m *SequenceShiftCalculatorOptions) GetEmitEmptyPacketsBeforeFirstPacket() bool {
	if m != nil && m.EmitEmptyPacketsBeforeFirstPacket != nil {
		return *m.EmitEmptyPacketsBeforeFirstPacket
	}
	return Default_SequenceShiftCalculatorOptions_EmitEmptyPacketsBeforeFirstPacket
}

var E_SequenceShiftCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*SequenceShiftCalculatorOptions)(nil),
	Field:         107633927,
	Name:          "mediapipe.SequenceShiftCalculatorOptions.ext",
	Tag:           "bytes,107633927,opt,name=ext",
	Filename:      "mediapipe/calculators/core/sequence_shift_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_SequenceShiftCalculatorOptions_Ext)
	proto.RegisterType((*SequenceShiftCalculatorOptions)(nil), "mediapipe.SequenceShiftCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/core/sequence_shift_calculator.proto", fileDescriptor_7e995a0b7788b71d)
}

var fileDescriptor_7e995a0b7788b71d = []byte{
	// 327 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x91, 0xb1, 0x4a, 0x03, 0x41,
	0x10, 0x86, 0x6f, 0x23, 0x01, 0x3d, 0xb5, 0xb9, 0x2a, 0x88, 0x0c, 0x51, 0x44, 0x63, 0xe1, 0x1d,
	0x46, 0xb0, 0x48, 0x19, 0xd1, 0x36, 0x92, 0x34, 0x22, 0xc2, 0x72, 0x39, 0x67, 0x93, 0x25, 0x77,
	0xd9, 0x75, 0x77, 0x83, 0xda, 0x69, 0x65, 0xeb, 0x63, 0xe8, 0x2b, 0xf8, 0x04, 0x96, 0x29, 0x53,
	0x9a, 0x4d, 0x63, 0x99, 0x47, 0x90, 0xcb, 0x85, 0x9c, 0x28, 0xda, 0xfe, 0xff, 0x7c, 0x1f, 0x3b,
	0x3b, 0x6e, 0x2d, 0xc1, 0x6b, 0x1e, 0x4a, 0x2e, 0x31, 0x88, 0xc2, 0x38, 0x1a, 0xc4, 0xa1, 0x11,
	0x4a, 0x07, 0x91, 0x50, 0x18, 0x68, 0xbc, 0x19, 0x60, 0x3f, 0x42, 0xaa, 0xbb, 0x9c, 0x19, 0x9a,
	0xf7, 0xbe, 0x54, 0xc2, 0x08, 0x6f, 0x65, 0xc1, 0x6e, 0xec, 0xe4, 0x1a, 0xa6, 0xc2, 0x04, 0x6f,
	0x85, 0xea, 0x05, 0x3f, 0x81, 0xed, 0xc7, 0x82, 0x0b, 0xad, 0xb9, 0xb4, 0x95, 0x3a, 0x4f, 0x16,
	0x13, 0x0d, 0x69, 0xb8, 0xe8, 0x6b, 0x6f, 0xcf, 0x5d, 0x97, 0x61, 0xd4, 0x43, 0x43, 0x05, 0x63,
	0x1a, 0x4d, 0x89, 0x94, 0x49, 0xa5, 0x58, 0x2b, 0x1c, 0x1c, 0x36, 0xd7, 0xb2, 0xa2, 0x31, 0xcb,
	0xbd, 0x0b, 0x77, 0x17, 0x13, 0x6e, 0x28, 0x26, 0xd2, 0xdc, 0xd3, 0xac, 0xd2, 0xb4, 0x8d, 0x4c,
	0x28, 0xa4, 0x8c, 0x2b, 0x6d, 0xe6, 0x61, 0xa9, 0x50, 0x26, 0x95, 0xe5, 0x5a, 0x91, 0x85, 0xb1,
	0xc6, 0xe6, 0x56, 0x0a, 0x9d, 0xa6, 0xcc, 0x79, 0x86, 0xd4, 0x67, 0xc4, 0x59, 0x0a, 0x64, 0x49,
	0xf5, 0xca, 0x5d, 0xc2, 0x3b, 0xe3, 0x6d, 0xfa, 0x8b, 0x9d, 0xfc, 0x5f, 0xef, 0x2c, 0x3d, 0xbd,
	0xbd, 0x1e, 0x95, 0x49, 0x65, 0xb5, 0xba, 0xff, 0x6d, 0xec, 0xff, 0xdd, 0x9a, 0xa9, 0xb6, 0x1e,
	0x0f, 0xc7, 0xe0, 0x8c, 0xc6, 0xe0, 0x4c, 0xc7, 0x40, 0x1e, 0x2c, 0x90, 0x17, 0x0b, 0xe4, 0xdd,
	0x02, 0x19, 0x5a, 0x20, 0x1f, 0x16, 0xc8, 0xa7, 0x05, 0x67, 0x6a, 0x81, 0x3c, 0x4f, 0xc0, 0x19,
	0x4e, 0xc0, 0x19, 0x4d, 0xc0, 0xb9, 0x3c, 0xee, 0x70, 0xd3, 0x1d, 0xb4, 0xfd, 0x48, 0x24, 0x41,
	0x47, 0x88, 0x4e, 0x8c, 0x41, 0xfe, 0xe3, 0x7f, 0x9f, 0xf0, 0x2b, 0x00, 0x00, 0xff, 0xff, 0xfb,
	0x10, 0x60, 0xda, 0xdf, 0x01, 0x00, 0x00,
}

func (this *SequenceShiftCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*SequenceShiftCalculatorOptions)
	if !ok {
		that2, ok := that.(SequenceShiftCalculatorOptions)
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
	if this.PacketOffset != nil && that1.PacketOffset != nil {
		if *this.PacketOffset != *that1.PacketOffset {
			return false
		}
	} else if this.PacketOffset != nil {
		return false
	} else if that1.PacketOffset != nil {
		return false
	}
	if this.EmitEmptyPacketsBeforeFirstPacket != nil && that1.EmitEmptyPacketsBeforeFirstPacket != nil {
		if *this.EmitEmptyPacketsBeforeFirstPacket != *that1.EmitEmptyPacketsBeforeFirstPacket {
			return false
		}
	} else if this.EmitEmptyPacketsBeforeFirstPacket != nil {
		return false
	} else if that1.EmitEmptyPacketsBeforeFirstPacket != nil {
		return false
	}
	return true
}
func (this *SequenceShiftCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&core.SequenceShiftCalculatorOptions{")
	if this.PacketOffset != nil {
		s = append(s, "PacketOffset: "+valueToGoStringSequenceShiftCalculator(this.PacketOffset, "int32")+",\n")
	}
	if this.EmitEmptyPacketsBeforeFirstPacket != nil {
		s = append(s, "EmitEmptyPacketsBeforeFirstPacket: "+valueToGoStringSequenceShiftCalculator(this.EmitEmptyPacketsBeforeFirstPacket, "bool")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringSequenceShiftCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *SequenceShiftCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SequenceShiftCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SequenceShiftCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.EmitEmptyPacketsBeforeFirstPacket != nil {
		i--
		if *m.EmitEmptyPacketsBeforeFirstPacket {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x10
	}
	if m.PacketOffset != nil {
		i = encodeVarintSequenceShiftCalculator(dAtA, i, uint64(*m.PacketOffset))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintSequenceShiftCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovSequenceShiftCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *SequenceShiftCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.PacketOffset != nil {
		n += 1 + sovSequenceShiftCalculator(uint64(*m.PacketOffset))
	}
	if m.EmitEmptyPacketsBeforeFirstPacket != nil {
		n += 2
	}
	return n
}

func sovSequenceShiftCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozSequenceShiftCalculator(x uint64) (n int) {
	return sovSequenceShiftCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *SequenceShiftCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&SequenceShiftCalculatorOptions{`,
		`PacketOffset:` + valueToStringSequenceShiftCalculator(this.PacketOffset) + `,`,
		`EmitEmptyPacketsBeforeFirstPacket:` + valueToStringSequenceShiftCalculator(this.EmitEmptyPacketsBeforeFirstPacket) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringSequenceShiftCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *SequenceShiftCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowSequenceShiftCalculator
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
			return fmt.Errorf("proto: SequenceShiftCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SequenceShiftCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PacketOffset", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSequenceShiftCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.PacketOffset = &v
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field EmitEmptyPacketsBeforeFirstPacket", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSequenceShiftCalculator
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
			m.EmitEmptyPacketsBeforeFirstPacket = &b
		default:
			iNdEx = preIndex
			skippy, err := skipSequenceShiftCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthSequenceShiftCalculator
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
func skipSequenceShiftCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowSequenceShiftCalculator
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
					return 0, ErrIntOverflowSequenceShiftCalculator
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
					return 0, ErrIntOverflowSequenceShiftCalculator
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
				return 0, ErrInvalidLengthSequenceShiftCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupSequenceShiftCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthSequenceShiftCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthSequenceShiftCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowSequenceShiftCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupSequenceShiftCalculator = fmt.Errorf("proto: unexpected end of group")
)