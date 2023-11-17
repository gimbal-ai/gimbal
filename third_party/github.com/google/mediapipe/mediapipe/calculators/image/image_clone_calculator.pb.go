// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/image/image_clone_calculator.proto

package image

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

type ImageCloneCalculatorOptions struct {
	OutputOnGpu *bool `protobuf:"varint,1,opt,name=output_on_gpu,json=outputOnGpu,def=0" json:"output_on_gpu,omitempty"`
}

func (m *ImageCloneCalculatorOptions) Reset()      { *m = ImageCloneCalculatorOptions{} }
func (*ImageCloneCalculatorOptions) ProtoMessage() {}
func (*ImageCloneCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_01d61ec489c8f915, []int{0}
}
func (m *ImageCloneCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ImageCloneCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ImageCloneCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ImageCloneCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ImageCloneCalculatorOptions.Merge(m, src)
}
func (m *ImageCloneCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *ImageCloneCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_ImageCloneCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_ImageCloneCalculatorOptions proto.InternalMessageInfo

const Default_ImageCloneCalculatorOptions_OutputOnGpu bool = false

func (m *ImageCloneCalculatorOptions) GetOutputOnGpu() bool {
	if m != nil && m.OutputOnGpu != nil {
		return *m.OutputOnGpu
	}
	return Default_ImageCloneCalculatorOptions_OutputOnGpu
}

var E_ImageCloneCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*ImageCloneCalculatorOptions)(nil),
	Field:         372781894,
	Name:          "mediapipe.ImageCloneCalculatorOptions.ext",
	Tag:           "bytes,372781894,opt,name=ext",
	Filename:      "mediapipe/calculators/image/image_clone_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_ImageCloneCalculatorOptions_Ext)
	proto.RegisterType((*ImageCloneCalculatorOptions)(nil), "mediapipe.ImageCloneCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/image/image_clone_calculator.proto", fileDescriptor_01d61ec489c8f915)
}

var fileDescriptor_01d61ec489c8f915 = []byte{
	// 266 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xb2, 0xc8, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0xcf, 0xcc, 0x4d, 0x4c, 0x4f, 0x85, 0x90, 0xf1, 0xc9, 0x39, 0xf9, 0x79, 0xa9, 0xf1,
	0x08, 0x59, 0xbd, 0x82, 0xa2, 0xfc, 0x92, 0x7c, 0x21, 0x4e, 0xb8, 0x4e, 0x29, 0x15, 0x84, 0x21,
	0x69, 0x45, 0x89, 0xb9, 0xa9, 0xe5, 0xf9, 0x45, 0xd9, 0xfa, 0xe8, 0x1a, 0x94, 0x96, 0x30, 0x72,
	0x49, 0x7b, 0x82, 0x4c, 0x74, 0x06, 0x19, 0xe8, 0x0c, 0x97, 0xf6, 0x2f, 0x28, 0xc9, 0xcc, 0xcf,
	0x2b, 0x16, 0xd2, 0xe4, 0xe2, 0xcd, 0x2f, 0x2d, 0x29, 0x28, 0x2d, 0x89, 0xcf, 0xcf, 0x8b, 0x4f,
	0x2f, 0x28, 0x95, 0x60, 0x54, 0x60, 0xd4, 0xe0, 0xb0, 0x62, 0x4d, 0x4b, 0xcc, 0x29, 0x4e, 0x0d,
	0xe2, 0x86, 0xc8, 0xf9, 0xe7, 0xb9, 0x17, 0x94, 0x1a, 0x45, 0x71, 0x31, 0xa7, 0x56, 0x94, 0x08,
	0xc9, 0xe8, 0xc1, 0x2d, 0xd6, 0xc3, 0x30, 0x4f, 0xe2, 0xd8, 0xb3, 0x07, 0x1b, 0x41, 0x46, 0x70,
	0x1b, 0xa9, 0x21, 0xa9, 0xc3, 0xe3, 0x82, 0x20, 0x90, 0xa1, 0x4e, 0xb9, 0x17, 0x1e, 0xca, 0x31,
	0xdc, 0x78, 0x28, 0xc7, 0xf0, 0xe1, 0xa1, 0x1c, 0x63, 0xc3, 0x23, 0x39, 0xc6, 0x15, 0x8f, 0xe4,
	0x18, 0x4f, 0x3c, 0x92, 0x63, 0xbc, 0xf0, 0x48, 0x8e, 0xf1, 0xc1, 0x23, 0x39, 0xc6, 0x17, 0x8f,
	0xe4, 0x18, 0x3e, 0x3c, 0x92, 0x63, 0x9c, 0xf0, 0x58, 0x8e, 0xe1, 0xc2, 0x63, 0x39, 0x86, 0x1b,
	0x8f, 0xe5, 0x18, 0xa2, 0xcc, 0xd3, 0x33, 0x4b, 0x32, 0x4a, 0x93, 0xf4, 0x92, 0xf3, 0x73, 0xf5,
	0xd3, 0xf3, 0xf3, 0xd3, 0x73, 0x52, 0xf5, 0x11, 0x81, 0x82, 0x27, 0x8c, 0x01, 0x01, 0x00, 0x00,
	0xff, 0xff, 0x72, 0x7c, 0xac, 0xc0, 0x81, 0x01, 0x00, 0x00,
}

func (this *ImageCloneCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ImageCloneCalculatorOptions)
	if !ok {
		that2, ok := that.(ImageCloneCalculatorOptions)
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
	if this.OutputOnGpu != nil && that1.OutputOnGpu != nil {
		if *this.OutputOnGpu != *that1.OutputOnGpu {
			return false
		}
	} else if this.OutputOnGpu != nil {
		return false
	} else if that1.OutputOnGpu != nil {
		return false
	}
	return true
}
func (this *ImageCloneCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&image.ImageCloneCalculatorOptions{")
	if this.OutputOnGpu != nil {
		s = append(s, "OutputOnGpu: "+valueToGoStringImageCloneCalculator(this.OutputOnGpu, "bool")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringImageCloneCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *ImageCloneCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ImageCloneCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ImageCloneCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.OutputOnGpu != nil {
		i--
		if *m.OutputOnGpu {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintImageCloneCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovImageCloneCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ImageCloneCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.OutputOnGpu != nil {
		n += 2
	}
	return n
}

func sovImageCloneCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozImageCloneCalculator(x uint64) (n int) {
	return sovImageCloneCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ImageCloneCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ImageCloneCalculatorOptions{`,
		`OutputOnGpu:` + valueToStringImageCloneCalculator(this.OutputOnGpu) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringImageCloneCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ImageCloneCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowImageCloneCalculator
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
			return fmt.Errorf("proto: ImageCloneCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ImageCloneCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutputOnGpu", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowImageCloneCalculator
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
			m.OutputOnGpu = &b
		default:
			iNdEx = preIndex
			skippy, err := skipImageCloneCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthImageCloneCalculator
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
func skipImageCloneCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowImageCloneCalculator
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
					return 0, ErrIntOverflowImageCloneCalculator
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
					return 0, ErrIntOverflowImageCloneCalculator
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
				return 0, ErrInvalidLengthImageCloneCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupImageCloneCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthImageCloneCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthImageCloneCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowImageCloneCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupImageCloneCalculator = fmt.Errorf("proto: unexpected end of group")
)