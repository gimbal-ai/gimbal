// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/common/typespb/uuid.proto

package typespb

import (
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

type UUID struct {
	HighBits uint64 `protobuf:"varint,1,opt,name=high_bits,json=highBits,proto3" json:"high_bits,omitempty"`
	LowBits  uint64 `protobuf:"varint,2,opt,name=low_bits,json=lowBits,proto3" json:"low_bits,omitempty"`
}

func (m *UUID) Reset()      { *m = UUID{} }
func (*UUID) ProtoMessage() {}
func (*UUID) Descriptor() ([]byte, []int) {
	return fileDescriptor_1933d1102ae7b01e, []int{0}
}
func (m *UUID) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *UUID) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_UUID.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *UUID) XXX_Merge(src proto.Message) {
	xxx_messageInfo_UUID.Merge(m, src)
}
func (m *UUID) XXX_Size() int {
	return m.Size()
}
func (m *UUID) XXX_DiscardUnknown() {
	xxx_messageInfo_UUID.DiscardUnknown(m)
}

var xxx_messageInfo_UUID proto.InternalMessageInfo

func (m *UUID) GetHighBits() uint64 {
	if m != nil {
		return m.HighBits
	}
	return 0
}

func (m *UUID) GetLowBits() uint64 {
	if m != nil {
		return m.LowBits
	}
	return 0
}

func init() {
	proto.RegisterType((*UUID)(nil), "gml.types.UUID")
}

func init() { proto.RegisterFile("src/common/typespb/uuid.proto", fileDescriptor_1933d1102ae7b01e) }

var fileDescriptor_1933d1102ae7b01e = []byte{
	// 201 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x92, 0x2d, 0x2e, 0x4a, 0xd6,
	0x4f, 0xce, 0xcf, 0xcd, 0xcd, 0xcf, 0xd3, 0x2f, 0xa9, 0x2c, 0x48, 0x2d, 0x2e, 0x48, 0xd2, 0x2f,
	0x2d, 0xcd, 0x4c, 0xd1, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0x4c, 0xcf, 0xcd, 0xd1, 0x03,
	0x8b, 0x2b, 0xd9, 0x71, 0xb1, 0x84, 0x86, 0x7a, 0xba, 0x08, 0x49, 0x73, 0x71, 0x66, 0x64, 0xa6,
	0x67, 0xc4, 0x27, 0x65, 0x96, 0x14, 0x4b, 0x30, 0x2a, 0x30, 0x6a, 0xb0, 0x04, 0x71, 0x80, 0x04,
	0x9c, 0x32, 0x4b, 0x8a, 0x85, 0x24, 0xb9, 0x38, 0x72, 0xf2, 0xcb, 0x21, 0x72, 0x4c, 0x60, 0x39,
	0xf6, 0x9c, 0xfc, 0x72, 0x90, 0x94, 0x53, 0xea, 0x85, 0x87, 0x72, 0x0c, 0x37, 0x1e, 0xca, 0x31,
	0x7c, 0x78, 0x28, 0xc7, 0xd8, 0xf0, 0x48, 0x8e, 0x71, 0xc5, 0x23, 0x39, 0xc6, 0x13, 0x8f, 0xe4,
	0x18, 0x2f, 0x3c, 0x92, 0x63, 0x7c, 0xf0, 0x48, 0x8e, 0xf1, 0xc5, 0x23, 0x39, 0x86, 0x0f, 0x8f,
	0xe4, 0x18, 0x27, 0x3c, 0x96, 0x63, 0xb8, 0xf0, 0x58, 0x8e, 0xe1, 0xc6, 0x63, 0x39, 0x86, 0x28,
	0xfd, 0xf4, 0xcc, 0xdc, 0x9c, 0xd4, 0x92, 0x9c, 0xc4, 0xa4, 0x62, 0xbd, 0xc4, 0x4c, 0x28, 0x4f,
	0x1f, 0xd3, 0xd1, 0xd6, 0x50, 0x3a, 0x89, 0x0d, 0xec, 0x70, 0x63, 0x40, 0x00, 0x00, 0x00, 0xff,
	0xff, 0xa8, 0x34, 0x56, 0x4f, 0xd9, 0x00, 0x00, 0x00,
}

func (this *UUID) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*UUID)
	if !ok {
		that2, ok := that.(UUID)
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
	if this.HighBits != that1.HighBits {
		return false
	}
	if this.LowBits != that1.LowBits {
		return false
	}
	return true
}
func (this *UUID) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&typespb.UUID{")
	s = append(s, "HighBits: "+fmt.Sprintf("%#v", this.HighBits)+",\n")
	s = append(s, "LowBits: "+fmt.Sprintf("%#v", this.LowBits)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringUuid(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *UUID) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *UUID) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *UUID) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.LowBits != 0 {
		i = encodeVarintUuid(dAtA, i, uint64(m.LowBits))
		i--
		dAtA[i] = 0x10
	}
	if m.HighBits != 0 {
		i = encodeVarintUuid(dAtA, i, uint64(m.HighBits))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintUuid(dAtA []byte, offset int, v uint64) int {
	offset -= sovUuid(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *UUID) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.HighBits != 0 {
		n += 1 + sovUuid(uint64(m.HighBits))
	}
	if m.LowBits != 0 {
		n += 1 + sovUuid(uint64(m.LowBits))
	}
	return n
}

func sovUuid(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozUuid(x uint64) (n int) {
	return sovUuid(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *UUID) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&UUID{`,
		`HighBits:` + fmt.Sprintf("%v", this.HighBits) + `,`,
		`LowBits:` + fmt.Sprintf("%v", this.LowBits) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringUuid(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *UUID) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowUuid
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
			return fmt.Errorf("proto: UUID: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: UUID: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field HighBits", wireType)
			}
			m.HighBits = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowUuid
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.HighBits |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field LowBits", wireType)
			}
			m.LowBits = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowUuid
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.LowBits |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipUuid(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthUuid
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
func skipUuid(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowUuid
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
					return 0, ErrIntOverflowUuid
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
					return 0, ErrIntOverflowUuid
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
				return 0, ErrInvalidLengthUuid
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupUuid
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthUuid
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthUuid        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowUuid          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupUuid = fmt.Errorf("proto: unexpected end of group")
)
