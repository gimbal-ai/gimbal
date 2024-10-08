// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/api/corepb/v1/gem_config.proto

package corepb

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

type GEMConfig struct {
}

func (m *GEMConfig) Reset()      { *m = GEMConfig{} }
func (*GEMConfig) ProtoMessage() {}
func (*GEMConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_b3a7eee323e19df1, []int{0}
}
func (m *GEMConfig) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *GEMConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_GEMConfig.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *GEMConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GEMConfig.Merge(m, src)
}
func (m *GEMConfig) XXX_Size() int {
	return m.Size()
}
func (m *GEMConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_GEMConfig.DiscardUnknown(m)
}

var xxx_messageInfo_GEMConfig proto.InternalMessageInfo

func init() {
	proto.RegisterType((*GEMConfig)(nil), "gml.internal.api.core.v1.GEMConfig")
}

func init() {
	proto.RegisterFile("src/api/corepb/v1/gem_config.proto", fileDescriptor_b3a7eee323e19df1)
}

var fileDescriptor_b3a7eee323e19df1 = []byte{
	// 180 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x52, 0x2a, 0x2e, 0x4a, 0xd6,
	0x4f, 0x2c, 0xc8, 0xd4, 0x4f, 0xce, 0x2f, 0x4a, 0x2d, 0x48, 0xd2, 0x2f, 0x33, 0xd4, 0x4f, 0x4f,
	0xcd, 0x8d, 0x4f, 0xce, 0xcf, 0x4b, 0xcb, 0x4c, 0xd7, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0x92,
	0x48, 0xcf, 0xcd, 0xd1, 0xcb, 0xcc, 0x2b, 0x49, 0x2d, 0xca, 0x4b, 0xcc, 0xd1, 0x4b, 0x2c, 0xc8,
	0xd4, 0x03, 0x29, 0xd6, 0x2b, 0x33, 0x54, 0xe2, 0xe6, 0xe2, 0x74, 0x77, 0xf5, 0x75, 0x06, 0x2b,
	0x76, 0x4a, 0xbe, 0xf0, 0x50, 0x8e, 0xe1, 0xc6, 0x43, 0x39, 0x86, 0x0f, 0x0f, 0xe5, 0x18, 0x1b,
	0x1e, 0xc9, 0x31, 0xae, 0x78, 0x24, 0xc7, 0x78, 0xe2, 0x91, 0x1c, 0xe3, 0x85, 0x47, 0x72, 0x8c,
	0x0f, 0x1e, 0xc9, 0x31, 0xbe, 0x78, 0x24, 0xc7, 0xf0, 0xe1, 0x91, 0x1c, 0xe3, 0x84, 0xc7, 0x72,
	0x0c, 0x17, 0x1e, 0xcb, 0x31, 0xdc, 0x78, 0x2c, 0xc7, 0x10, 0xa5, 0x9b, 0x9e, 0x99, 0x9b, 0x93,
	0x5a, 0x92, 0x93, 0x98, 0x54, 0xac, 0x97, 0x98, 0xa9, 0x0f, 0xe1, 0xe9, 0x63, 0xb8, 0xca, 0x1a,
	0xc2, 0x4a, 0x62, 0x03, 0x3b, 0xc9, 0x18, 0x10, 0x00, 0x00, 0xff, 0xff, 0x95, 0x8b, 0x6d, 0xfd,
	0xb8, 0x00, 0x00, 0x00,
}

func (this *GEMConfig) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*GEMConfig)
	if !ok {
		that2, ok := that.(GEMConfig)
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
func (this *GEMConfig) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 4)
	s = append(s, "&corepb.GEMConfig{")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringGemConfig(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *GEMConfig) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *GEMConfig) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *GEMConfig) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	return len(dAtA) - i, nil
}

func encodeVarintGemConfig(dAtA []byte, offset int, v uint64) int {
	offset -= sovGemConfig(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *GEMConfig) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	return n
}

func sovGemConfig(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozGemConfig(x uint64) (n int) {
	return sovGemConfig(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *GEMConfig) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&GEMConfig{`,
		`}`,
	}, "")
	return s
}
func valueToStringGemConfig(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *GEMConfig) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowGemConfig
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
			return fmt.Errorf("proto: GEMConfig: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: GEMConfig: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		default:
			iNdEx = preIndex
			skippy, err := skipGemConfig(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthGemConfig
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
func skipGemConfig(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowGemConfig
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
					return 0, ErrIntOverflowGemConfig
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
					return 0, ErrIntOverflowGemConfig
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
				return 0, ErrInvalidLengthGemConfig
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupGemConfig
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthGemConfig
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthGemConfig        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowGemConfig          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupGemConfig = fmt.Errorf("proto: unexpected end of group")
)
