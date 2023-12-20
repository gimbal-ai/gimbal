// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/calculators/core/optionspb/packet_latency_metrics_sink_calculator_options.proto

package optionspb

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

type PacketLatencyMetricsSinkCalculatorOptions struct {
	Name string `protobuf:"bytes,1,opt,name=name,proto3" json:"name,omitempty"`
}

func (m *PacketLatencyMetricsSinkCalculatorOptions) Reset() {
	*m = PacketLatencyMetricsSinkCalculatorOptions{}
}
func (*PacketLatencyMetricsSinkCalculatorOptions) ProtoMessage() {}
func (*PacketLatencyMetricsSinkCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_1fadef9e8d59af86, []int{0}
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_PacketLatencyMetricsSinkCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_PacketLatencyMetricsSinkCalculatorOptions.Merge(m, src)
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_PacketLatencyMetricsSinkCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_PacketLatencyMetricsSinkCalculatorOptions proto.InternalMessageInfo

func (m *PacketLatencyMetricsSinkCalculatorOptions) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func init() {
	proto.RegisterType((*PacketLatencyMetricsSinkCalculatorOptions)(nil), "gml.gem.calculators.core.optionspb.PacketLatencyMetricsSinkCalculatorOptions")
}

func init() {
	proto.RegisterFile("src/gem/calculators/core/optionspb/packet_latency_metrics_sink_calculator_options.proto", fileDescriptor_1fadef9e8d59af86)
}

var fileDescriptor_1fadef9e8d59af86 = []byte{
	// 242 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0xd0, 0xbf, 0x4a, 0xc7, 0x30,
	0x10, 0x07, 0xf0, 0x1c, 0x88, 0x60, 0xc7, 0x4e, 0x4e, 0x87, 0xfc, 0x26, 0x5d, 0x92, 0xc1, 0xd1,
	0x41, 0xd4, 0x55, 0x51, 0x74, 0x10, 0x5c, 0x42, 0x1a, 0x42, 0x09, 0xcd, 0x9f, 0x92, 0x44, 0xc4,
	0xcd, 0x47, 0xf0, 0x31, 0x7c, 0x14, 0xc7, 0x8e, 0xbf, 0xd1, 0xa6, 0x8b, 0x63, 0x1f, 0x41, 0x6c,
	0xa4, 0xba, 0xb9, 0x7d, 0x0f, 0xee, 0xfb, 0x81, 0xbb, 0xea, 0x3e, 0x06, 0xc9, 0x5a, 0x65, 0x99,
	0x14, 0x46, 0x3e, 0x1a, 0x91, 0x7c, 0x88, 0x4c, 0xfa, 0xa0, 0x98, 0xef, 0x93, 0xf6, 0x2e, 0xf6,
	0x0d, 0xeb, 0x85, 0xec, 0x54, 0xe2, 0x46, 0x24, 0xe5, 0xe4, 0x33, 0xb7, 0x2a, 0x05, 0x2d, 0x23,
	0x8f, 0xda, 0x75, 0xfc, 0xb7, 0xc6, 0x7f, 0x0a, 0xb4, 0x0f, 0x3e, 0xf9, 0x7a, 0xd3, 0x5a, 0x43,
	0x5b, 0x65, 0xe9, 0x1f, 0x98, 0x7e, 0xc3, 0x74, 0x85, 0x37, 0xa7, 0xd5, 0xd1, 0xcd, 0x62, 0x5f,
	0x16, 0xfa, 0xaa, 0xc8, 0x77, 0xda, 0x75, 0x17, 0x6b, 0xeb, 0xba, 0xac, 0xd7, 0x75, 0xb5, 0xe3,
	0x84, 0x55, 0xfb, 0x70, 0x00, 0x87, 0x7b, 0xb7, 0x4b, 0x3e, 0x7f, 0x1a, 0x46, 0x24, 0xdb, 0x11,
	0xc9, 0x3c, 0x22, 0xbc, 0x64, 0x84, 0xb7, 0x8c, 0xf0, 0x9e, 0x11, 0x86, 0x8c, 0xf0, 0x91, 0x11,
	0x3e, 0x33, 0x92, 0x39, 0x23, 0xbc, 0x4e, 0x48, 0x86, 0x09, 0xc9, 0x76, 0x42, 0xf2, 0x70, 0xd6,
	0x6a, 0x6b, 0x54, 0x32, 0xa2, 0x89, 0x54, 0x68, 0x56, 0x26, 0xf6, 0xff, 0x23, 0x4e, 0xd6, 0xd4,
	0xec, 0x2e, 0x47, 0x1e, 0x7f, 0x05, 0x00, 0x00, 0xff, 0xff, 0xce, 0xb3, 0xe9, 0xc7, 0x3f, 0x01,
	0x00, 0x00,
}

func (this *PacketLatencyMetricsSinkCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*PacketLatencyMetricsSinkCalculatorOptions)
	if !ok {
		that2, ok := that.(PacketLatencyMetricsSinkCalculatorOptions)
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
	if this.Name != that1.Name {
		return false
	}
	return true
}
func (this *PacketLatencyMetricsSinkCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&optionspb.PacketLatencyMetricsSinkCalculatorOptions{")
	s = append(s, "Name: "+fmt.Sprintf("%#v", this.Name)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringPacketLatencyMetricsSinkCalculatorOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *PacketLatencyMetricsSinkCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *PacketLatencyMetricsSinkCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Name) > 0 {
		i -= len(m.Name)
		copy(dAtA[i:], m.Name)
		i = encodeVarintPacketLatencyMetricsSinkCalculatorOptions(dAtA, i, uint64(len(m.Name)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintPacketLatencyMetricsSinkCalculatorOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovPacketLatencyMetricsSinkCalculatorOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovPacketLatencyMetricsSinkCalculatorOptions(uint64(l))
	}
	return n
}

func sovPacketLatencyMetricsSinkCalculatorOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozPacketLatencyMetricsSinkCalculatorOptions(x uint64) (n int) {
	return sovPacketLatencyMetricsSinkCalculatorOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *PacketLatencyMetricsSinkCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&PacketLatencyMetricsSinkCalculatorOptions{`,
		`Name:` + fmt.Sprintf("%v", this.Name) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringPacketLatencyMetricsSinkCalculatorOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *PacketLatencyMetricsSinkCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowPacketLatencyMetricsSinkCalculatorOptions
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
			return fmt.Errorf("proto: PacketLatencyMetricsSinkCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: PacketLatencyMetricsSinkCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowPacketLatencyMetricsSinkCalculatorOptions
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
				return ErrInvalidLengthPacketLatencyMetricsSinkCalculatorOptions
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthPacketLatencyMetricsSinkCalculatorOptions
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipPacketLatencyMetricsSinkCalculatorOptions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthPacketLatencyMetricsSinkCalculatorOptions
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
func skipPacketLatencyMetricsSinkCalculatorOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowPacketLatencyMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowPacketLatencyMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowPacketLatencyMetricsSinkCalculatorOptions
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
				return 0, ErrInvalidLengthPacketLatencyMetricsSinkCalculatorOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupPacketLatencyMetricsSinkCalculatorOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthPacketLatencyMetricsSinkCalculatorOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthPacketLatencyMetricsSinkCalculatorOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowPacketLatencyMetricsSinkCalculatorOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupPacketLatencyMetricsSinkCalculatorOptions = fmt.Errorf("proto: unexpected end of group")
)