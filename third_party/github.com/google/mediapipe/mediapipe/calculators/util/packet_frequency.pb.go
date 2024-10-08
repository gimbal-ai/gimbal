// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/util/packet_frequency.proto

package util

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

type PacketFrequency struct {
	PacketFrequencyHz float64 `protobuf:"fixed64,1,opt,name=packet_frequency_hz,json=packetFrequencyHz" json:"packet_frequency_hz"`
	Label             string  `protobuf:"bytes,2,opt,name=label" json:"label"`
}

func (m *PacketFrequency) Reset()      { *m = PacketFrequency{} }
func (*PacketFrequency) ProtoMessage() {}
func (*PacketFrequency) Descriptor() ([]byte, []int) {
	return fileDescriptor_1020b72cda34bcaa, []int{0}
}
func (m *PacketFrequency) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *PacketFrequency) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_PacketFrequency.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *PacketFrequency) XXX_Merge(src proto.Message) {
	xxx_messageInfo_PacketFrequency.Merge(m, src)
}
func (m *PacketFrequency) XXX_Size() int {
	return m.Size()
}
func (m *PacketFrequency) XXX_DiscardUnknown() {
	xxx_messageInfo_PacketFrequency.DiscardUnknown(m)
}

var xxx_messageInfo_PacketFrequency proto.InternalMessageInfo

func (m *PacketFrequency) GetPacketFrequencyHz() float64 {
	if m != nil {
		return m.PacketFrequencyHz
	}
	return 0
}

func (m *PacketFrequency) GetLabel() string {
	if m != nil {
		return m.Label
	}
	return ""
}

func init() {
	proto.RegisterType((*PacketFrequency)(nil), "mediapipe.PacketFrequency")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/util/packet_frequency.proto", fileDescriptor_1020b72cda34bcaa)
}

var fileDescriptor_1020b72cda34bcaa = []byte{
	// 218 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x32, 0xcc, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0x2f, 0x2d, 0xc9, 0xcc, 0xd1, 0x2f, 0x48, 0x4c, 0xce, 0x4e, 0x2d, 0x89, 0x4f, 0x2b,
	0x4a, 0x2d, 0x2c, 0x4d, 0xcd, 0x4b, 0xae, 0xd4, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0x84,
	0x6b, 0x51, 0x4a, 0xe6, 0xe2, 0x0f, 0x00, 0x2b, 0x72, 0x83, 0xa9, 0x11, 0x32, 0xe1, 0x12, 0x46,
	0xd7, 0x17, 0x9f, 0x51, 0x25, 0xc1, 0xa8, 0xc0, 0xa8, 0xc1, 0xe8, 0xc4, 0x72, 0xe2, 0x9e, 0x3c,
	0x43, 0x90, 0x60, 0x01, 0xaa, 0x1e, 0x8f, 0x2a, 0x21, 0x29, 0x2e, 0xd6, 0x9c, 0xc4, 0xa4, 0xd4,
	0x1c, 0x09, 0x26, 0x05, 0x46, 0x0d, 0x4e, 0xa8, 0x3a, 0x88, 0x90, 0x53, 0xce, 0x85, 0x87, 0x72,
	0x0c, 0x37, 0x1e, 0xca, 0x31, 0x7c, 0x78, 0x28, 0xc7, 0xd8, 0xf0, 0x48, 0x8e, 0x71, 0xc5, 0x23,
	0x39, 0xc6, 0x13, 0x8f, 0xe4, 0x18, 0x2f, 0x3c, 0x92, 0x63, 0x7c, 0xf0, 0x48, 0x8e, 0xf1, 0xc5,
	0x23, 0x39, 0x86, 0x0f, 0x8f, 0xe4, 0x18, 0x27, 0x3c, 0x96, 0x63, 0xb8, 0xf0, 0x58, 0x8e, 0xe1,
	0xc6, 0x63, 0x39, 0x86, 0x28, 0xb3, 0xf4, 0xcc, 0x92, 0x8c, 0xd2, 0x24, 0xbd, 0xe4, 0xfc, 0x5c,
	0xfd, 0xf4, 0xfc, 0xfc, 0xf4, 0x9c, 0x54, 0x7d, 0x84, 0x57, 0x71, 0x7b, 0x1a, 0x10, 0x00, 0x00,
	0xff, 0xff, 0xfa, 0x15, 0xec, 0xc9, 0x11, 0x01, 0x00, 0x00,
}

func (this *PacketFrequency) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*PacketFrequency)
	if !ok {
		that2, ok := that.(PacketFrequency)
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
	if this.PacketFrequencyHz != that1.PacketFrequencyHz {
		return false
	}
	if this.Label != that1.Label {
		return false
	}
	return true
}
func (this *PacketFrequency) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&util.PacketFrequency{")
	s = append(s, "PacketFrequencyHz: "+fmt.Sprintf("%#v", this.PacketFrequencyHz)+",\n")
	s = append(s, "Label: "+fmt.Sprintf("%#v", this.Label)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringPacketFrequency(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *PacketFrequency) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *PacketFrequency) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *PacketFrequency) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= len(m.Label)
	copy(dAtA[i:], m.Label)
	i = encodeVarintPacketFrequency(dAtA, i, uint64(len(m.Label)))
	i--
	dAtA[i] = 0x12
	i -= 8
	encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.PacketFrequencyHz))))
	i--
	dAtA[i] = 0x9
	return len(dAtA) - i, nil
}

func encodeVarintPacketFrequency(dAtA []byte, offset int, v uint64) int {
	offset -= sovPacketFrequency(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *PacketFrequency) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 9
	l = len(m.Label)
	n += 1 + l + sovPacketFrequency(uint64(l))
	return n
}

func sovPacketFrequency(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozPacketFrequency(x uint64) (n int) {
	return sovPacketFrequency(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *PacketFrequency) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&PacketFrequency{`,
		`PacketFrequencyHz:` + fmt.Sprintf("%v", this.PacketFrequencyHz) + `,`,
		`Label:` + fmt.Sprintf("%v", this.Label) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringPacketFrequency(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *PacketFrequency) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowPacketFrequency
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
			return fmt.Errorf("proto: PacketFrequency: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: PacketFrequency: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field PacketFrequencyHz", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.PacketFrequencyHz = float64(math.Float64frombits(v))
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Label", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowPacketFrequency
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthPacketFrequency
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthPacketFrequency
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Label = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipPacketFrequency(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthPacketFrequency
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
func skipPacketFrequency(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowPacketFrequency
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
					return 0, ErrIntOverflowPacketFrequency
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
					return 0, ErrIntOverflowPacketFrequency
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
				return 0, ErrInvalidLengthPacketFrequency
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupPacketFrequency
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthPacketFrequency
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthPacketFrequency        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowPacketFrequency          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupPacketFrequency = fmt.Errorf("proto: unexpected end of group")
)
