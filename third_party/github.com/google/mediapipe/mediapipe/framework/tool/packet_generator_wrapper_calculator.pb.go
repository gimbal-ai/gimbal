// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/tool/packet_generator_wrapper_calculator.proto

package tool

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

type PacketGeneratorWrapperCalculatorOptions struct {
	PacketGenerator string                            `protobuf:"bytes,1,opt,name=packet_generator,json=packetGenerator" json:"packet_generator"`
	Options         *framework.PacketGeneratorOptions `protobuf:"bytes,2,opt,name=options" json:"options,omitempty"`
	Package         string                            `protobuf:"bytes,3,opt,name=package" json:"package"`
}

func (m *PacketGeneratorWrapperCalculatorOptions) Reset() {
	*m = PacketGeneratorWrapperCalculatorOptions{}
}
func (*PacketGeneratorWrapperCalculatorOptions) ProtoMessage() {}
func (*PacketGeneratorWrapperCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_03810df11d84234f, []int{0}
}
func (m *PacketGeneratorWrapperCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *PacketGeneratorWrapperCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_PacketGeneratorWrapperCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *PacketGeneratorWrapperCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_PacketGeneratorWrapperCalculatorOptions.Merge(m, src)
}
func (m *PacketGeneratorWrapperCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *PacketGeneratorWrapperCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_PacketGeneratorWrapperCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_PacketGeneratorWrapperCalculatorOptions proto.InternalMessageInfo

func (m *PacketGeneratorWrapperCalculatorOptions) GetPacketGenerator() string {
	if m != nil {
		return m.PacketGenerator
	}
	return ""
}

func (m *PacketGeneratorWrapperCalculatorOptions) GetOptions() *framework.PacketGeneratorOptions {
	if m != nil {
		return m.Options
	}
	return nil
}

func (m *PacketGeneratorWrapperCalculatorOptions) GetPackage() string {
	if m != nil {
		return m.Package
	}
	return ""
}

var E_PacketGeneratorWrapperCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*PacketGeneratorWrapperCalculatorOptions)(nil),
	Field:         381945445,
	Name:          "mediapipe.PacketGeneratorWrapperCalculatorOptions.ext",
	Tag:           "bytes,381945445,opt,name=ext",
	Filename:      "mediapipe/framework/tool/packet_generator_wrapper_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_PacketGeneratorWrapperCalculatorOptions_Ext)
	proto.RegisterType((*PacketGeneratorWrapperCalculatorOptions)(nil), "mediapipe.PacketGeneratorWrapperCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/framework/tool/packet_generator_wrapper_calculator.proto", fileDescriptor_03810df11d84234f)
}

var fileDescriptor_03810df11d84234f = []byte{
	// 318 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x72, 0xca, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x2b, 0x4a, 0xcc, 0x4d, 0x2d, 0xcf, 0x2f, 0xca, 0xd6,
	0x2f, 0xc9, 0xcf, 0xcf, 0xd1, 0x2f, 0x48, 0x4c, 0xce, 0x4e, 0x2d, 0x89, 0x4f, 0x4f, 0xcd, 0x4b,
	0x2d, 0x4a, 0x2c, 0xc9, 0x2f, 0x8a, 0x2f, 0x2f, 0x4a, 0x2c, 0x28, 0x48, 0x2d, 0x8a, 0x4f, 0x4e,
	0xcc, 0x49, 0x2e, 0xcd, 0x01, 0x09, 0xe9, 0x15, 0x14, 0xe5, 0x97, 0xe4, 0x0b, 0x71, 0xc2, 0xcd,
	0x90, 0xd2, 0xc1, 0x66, 0x1c, 0x42, 0x43, 0x7c, 0x7e, 0x41, 0x49, 0x66, 0x7e, 0x5e, 0x31, 0x44,
	0xa3, 0x94, 0x16, 0x36, 0xd5, 0xe8, 0xf6, 0x42, 0xd4, 0x2a, 0xcd, 0x67, 0xe2, 0x52, 0x0f, 0x00,
	0x4b, 0xb9, 0xc3, 0x64, 0xc2, 0x21, 0x0e, 0x72, 0x86, 0x1b, 0xef, 0x0f, 0x31, 0x5d, 0x48, 0x9f,
	0x4b, 0x00, 0xdd, 0x14, 0x09, 0x46, 0x05, 0x46, 0x0d, 0x4e, 0x27, 0x96, 0x13, 0xf7, 0xe4, 0x19,
	0x82, 0xf8, 0x0b, 0x50, 0x0d, 0x12, 0xb2, 0xe6, 0x62, 0x87, 0xba, 0x4c, 0x82, 0x49, 0x81, 0x51,
	0x83, 0xdb, 0x48, 0x51, 0x0f, 0xee, 0x34, 0x3d, 0x34, 0x5b, 0xa1, 0x96, 0x04, 0xc1, 0x74, 0x08,
	0xc9, 0x71, 0xb1, 0x83, 0xcc, 0x4b, 0x4c, 0x4f, 0x95, 0x60, 0x46, 0xb2, 0x04, 0x26, 0x68, 0x94,
	0xc6, 0xc5, 0x9c, 0x5a, 0x51, 0x22, 0x24, 0x83, 0x64, 0x24, 0x86, 0x93, 0x25, 0x9e, 0xf6, 0x4c,
	0xd8, 0xc6, 0x08, 0xb6, 0xda, 0x08, 0xb7, 0xd5, 0xb8, 0x3c, 0x1c, 0x04, 0xb2, 0xc0, 0x29, 0xeb,
	0xc2, 0x43, 0x39, 0x86, 0x1b, 0x0f, 0xe5, 0x18, 0x3e, 0x3c, 0x94, 0x63, 0x6c, 0x78, 0x24, 0xc7,
	0xb8, 0xe2, 0x91, 0x1c, 0xe3, 0x89, 0x47, 0x72, 0x8c, 0x17, 0x1e, 0xc9, 0x31, 0x3e, 0x78, 0x24,
	0xc7, 0xf8, 0xe2, 0x91, 0x1c, 0xc3, 0x87, 0x47, 0x72, 0x8c, 0x13, 0x1e, 0xcb, 0x31, 0x5c, 0x78,
	0x2c, 0xc7, 0x70, 0xe3, 0xb1, 0x1c, 0x43, 0x94, 0x49, 0x7a, 0x66, 0x49, 0x46, 0x69, 0x92, 0x5e,
	0x72, 0x7e, 0xae, 0x7e, 0x7a, 0x7e, 0x7e, 0x7a, 0x4e, 0xaa, 0x3e, 0x22, 0x56, 0x70, 0x25, 0x0e,
	0x40, 0x00, 0x00, 0x00, 0xff, 0xff, 0x84, 0xf4, 0x51, 0xbd, 0x37, 0x02, 0x00, 0x00,
}

func (this *PacketGeneratorWrapperCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*PacketGeneratorWrapperCalculatorOptions)
	if !ok {
		that2, ok := that.(PacketGeneratorWrapperCalculatorOptions)
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
	if this.PacketGenerator != that1.PacketGenerator {
		return false
	}
	if !this.Options.Equal(that1.Options) {
		return false
	}
	if this.Package != that1.Package {
		return false
	}
	return true
}
func (this *PacketGeneratorWrapperCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&tool.PacketGeneratorWrapperCalculatorOptions{")
	s = append(s, "PacketGenerator: "+fmt.Sprintf("%#v", this.PacketGenerator)+",\n")
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	s = append(s, "Package: "+fmt.Sprintf("%#v", this.Package)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringPacketGeneratorWrapperCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *PacketGeneratorWrapperCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *PacketGeneratorWrapperCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *PacketGeneratorWrapperCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= len(m.Package)
	copy(dAtA[i:], m.Package)
	i = encodeVarintPacketGeneratorWrapperCalculator(dAtA, i, uint64(len(m.Package)))
	i--
	dAtA[i] = 0x1a
	if m.Options != nil {
		{
			size, err := m.Options.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintPacketGeneratorWrapperCalculator(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	i -= len(m.PacketGenerator)
	copy(dAtA[i:], m.PacketGenerator)
	i = encodeVarintPacketGeneratorWrapperCalculator(dAtA, i, uint64(len(m.PacketGenerator)))
	i--
	dAtA[i] = 0xa
	return len(dAtA) - i, nil
}

func encodeVarintPacketGeneratorWrapperCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovPacketGeneratorWrapperCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *PacketGeneratorWrapperCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.PacketGenerator)
	n += 1 + l + sovPacketGeneratorWrapperCalculator(uint64(l))
	if m.Options != nil {
		l = m.Options.Size()
		n += 1 + l + sovPacketGeneratorWrapperCalculator(uint64(l))
	}
	l = len(m.Package)
	n += 1 + l + sovPacketGeneratorWrapperCalculator(uint64(l))
	return n
}

func sovPacketGeneratorWrapperCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozPacketGeneratorWrapperCalculator(x uint64) (n int) {
	return sovPacketGeneratorWrapperCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *PacketGeneratorWrapperCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&PacketGeneratorWrapperCalculatorOptions{`,
		`PacketGenerator:` + fmt.Sprintf("%v", this.PacketGenerator) + `,`,
		`Options:` + strings.Replace(fmt.Sprintf("%v", this.Options), "PacketGeneratorOptions", "framework.PacketGeneratorOptions", 1) + `,`,
		`Package:` + fmt.Sprintf("%v", this.Package) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringPacketGeneratorWrapperCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *PacketGeneratorWrapperCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowPacketGeneratorWrapperCalculator
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
			return fmt.Errorf("proto: PacketGeneratorWrapperCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: PacketGeneratorWrapperCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field PacketGenerator", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowPacketGeneratorWrapperCalculator
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
				return ErrInvalidLengthPacketGeneratorWrapperCalculator
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthPacketGeneratorWrapperCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.PacketGenerator = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Options", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowPacketGeneratorWrapperCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				msglen |= int(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			if msglen < 0 {
				return ErrInvalidLengthPacketGeneratorWrapperCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthPacketGeneratorWrapperCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Options == nil {
				m.Options = &framework.PacketGeneratorOptions{}
			}
			if err := m.Options.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Package", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowPacketGeneratorWrapperCalculator
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
				return ErrInvalidLengthPacketGeneratorWrapperCalculator
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthPacketGeneratorWrapperCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Package = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipPacketGeneratorWrapperCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthPacketGeneratorWrapperCalculator
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
func skipPacketGeneratorWrapperCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowPacketGeneratorWrapperCalculator
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
					return 0, ErrIntOverflowPacketGeneratorWrapperCalculator
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
					return 0, ErrIntOverflowPacketGeneratorWrapperCalculator
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
				return 0, ErrInvalidLengthPacketGeneratorWrapperCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupPacketGeneratorWrapperCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthPacketGeneratorWrapperCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthPacketGeneratorWrapperCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowPacketGeneratorWrapperCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupPacketGeneratorWrapperCalculator = fmt.Errorf("proto: unexpected end of group")
)
