// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/calculators/plugin/cpu_tensor/optionspb/segmentation_masks_to_proto_options.proto

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

type SegmentationMasksToProtoOptions struct {
	IndexToLabel []string `protobuf:"bytes,1,rep,name=index_to_label,json=indexToLabel,proto3" json:"index_to_label,omitempty"`
}

func (m *SegmentationMasksToProtoOptions) Reset()      { *m = SegmentationMasksToProtoOptions{} }
func (*SegmentationMasksToProtoOptions) ProtoMessage() {}
func (*SegmentationMasksToProtoOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_9c48ea3a6ee2c3c7, []int{0}
}
func (m *SegmentationMasksToProtoOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SegmentationMasksToProtoOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_SegmentationMasksToProtoOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *SegmentationMasksToProtoOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SegmentationMasksToProtoOptions.Merge(m, src)
}
func (m *SegmentationMasksToProtoOptions) XXX_Size() int {
	return m.Size()
}
func (m *SegmentationMasksToProtoOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_SegmentationMasksToProtoOptions.DiscardUnknown(m)
}

var xxx_messageInfo_SegmentationMasksToProtoOptions proto.InternalMessageInfo

func (m *SegmentationMasksToProtoOptions) GetIndexToLabel() []string {
	if m != nil {
		return m.IndexToLabel
	}
	return nil
}

func init() {
	proto.RegisterType((*SegmentationMasksToProtoOptions)(nil), "gml.gem.calculators.cpu_tensor.optionspb.SegmentationMasksToProtoOptions")
}

func init() {
	proto.RegisterFile("src/gem/calculators/plugin/cpu_tensor/optionspb/segmentation_masks_to_proto_options.proto", fileDescriptor_9c48ea3a6ee2c3c7)
}

var fileDescriptor_9c48ea3a6ee2c3c7 = []byte{
	// 259 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x90, 0xbf, 0x4a, 0x34, 0x31,
	0x14, 0xc5, 0x13, 0x3e, 0xf8, 0xc0, 0x41, 0x2c, 0xb6, 0xb2, 0xba, 0x8a, 0x58, 0x6c, 0x95, 0x14,
	0x96, 0x76, 0x36, 0x36, 0xfe, 0x43, 0xb7, 0xd1, 0x26, 0x64, 0xc6, 0x10, 0x82, 0x49, 0x6e, 0x98,
	0x64, 0xc0, 0xd2, 0xc2, 0x07, 0xf0, 0x31, 0x7c, 0x14, 0xcb, 0x29, 0xb7, 0x74, 0x32, 0x8d, 0xe5,
	0x3e, 0x82, 0x64, 0x56, 0xd6, 0x6d, 0xed, 0xee, 0xef, 0x70, 0xf8, 0x5d, 0x38, 0xd5, 0x7d, 0x6c,
	0x1b, 0xae, 0x95, 0xe3, 0x8d, 0xb4, 0x4d, 0x67, 0x65, 0xc2, 0x36, 0xf2, 0x60, 0x3b, 0x6d, 0x3c,
	0x6f, 0x42, 0x27, 0x92, 0xf2, 0x11, 0x5b, 0x8e, 0x21, 0x19, 0xf4, 0x31, 0xd4, 0x3c, 0x2a, 0xed,
	0x94, 0x4f, 0xb2, 0xb0, 0x70, 0x32, 0x3e, 0x45, 0x91, 0x50, 0x84, 0x16, 0x13, 0x8a, 0x9f, 0x16,
	0x9b, 0x68, 0x36, 0xd7, 0xce, 0x32, 0xad, 0x1c, 0xdb, 0x52, 0xb3, 0x5f, 0x27, 0xdb, 0x38, 0x8f,
	0xce, 0xab, 0x83, 0xbb, 0x2d, 0xed, 0x65, 0xb1, 0x2e, 0xf0, 0xa6, 0x58, 0xae, 0xd7, 0xa5, 0xd9,
	0x71, 0xb5, 0x67, 0xfc, 0xa3, 0x7a, 0x2e, 0xcf, 0xac, 0xac, 0x95, 0xdd, 0xa7, 0x87, 0xff, 0xe6,
	0x3b, 0xb7, 0xbb, 0x53, 0xba, 0xc0, 0x8b, 0x92, 0x9d, 0xbd, 0xd2, 0x7e, 0x00, 0xb2, 0x1c, 0x80,
	0xac, 0x06, 0xa0, 0x2f, 0x19, 0xe8, 0x7b, 0x06, 0xfa, 0x91, 0x81, 0xf6, 0x19, 0xe8, 0x67, 0x06,
	0xfa, 0x95, 0x81, 0xac, 0x32, 0xd0, 0xb7, 0x11, 0x48, 0x3f, 0x02, 0x59, 0x8e, 0x40, 0x1e, 0xae,
	0xb4, 0x71, 0x56, 0x25, 0x2b, 0xeb, 0xc8, 0xa4, 0xe1, 0x6b, 0xe2, 0x7f, 0x5c, 0xe6, 0x74, 0x73,
	0xd5, 0xff, 0xa7, 0x01, 0x4e, 0xbe, 0x03, 0x00, 0x00, 0xff, 0xff, 0x3c, 0x4d, 0xb5, 0x04, 0x5d,
	0x01, 0x00, 0x00,
}

func (this *SegmentationMasksToProtoOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*SegmentationMasksToProtoOptions)
	if !ok {
		that2, ok := that.(SegmentationMasksToProtoOptions)
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
	if len(this.IndexToLabel) != len(that1.IndexToLabel) {
		return false
	}
	for i := range this.IndexToLabel {
		if this.IndexToLabel[i] != that1.IndexToLabel[i] {
			return false
		}
	}
	return true
}
func (this *SegmentationMasksToProtoOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&optionspb.SegmentationMasksToProtoOptions{")
	s = append(s, "IndexToLabel: "+fmt.Sprintf("%#v", this.IndexToLabel)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringSegmentationMasksToProtoOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *SegmentationMasksToProtoOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SegmentationMasksToProtoOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SegmentationMasksToProtoOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.IndexToLabel) > 0 {
		for iNdEx := len(m.IndexToLabel) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.IndexToLabel[iNdEx])
			copy(dAtA[i:], m.IndexToLabel[iNdEx])
			i = encodeVarintSegmentationMasksToProtoOptions(dAtA, i, uint64(len(m.IndexToLabel[iNdEx])))
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintSegmentationMasksToProtoOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovSegmentationMasksToProtoOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *SegmentationMasksToProtoOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.IndexToLabel) > 0 {
		for _, s := range m.IndexToLabel {
			l = len(s)
			n += 1 + l + sovSegmentationMasksToProtoOptions(uint64(l))
		}
	}
	return n
}

func sovSegmentationMasksToProtoOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozSegmentationMasksToProtoOptions(x uint64) (n int) {
	return sovSegmentationMasksToProtoOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *SegmentationMasksToProtoOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&SegmentationMasksToProtoOptions{`,
		`IndexToLabel:` + fmt.Sprintf("%v", this.IndexToLabel) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringSegmentationMasksToProtoOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *SegmentationMasksToProtoOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowSegmentationMasksToProtoOptions
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
			return fmt.Errorf("proto: SegmentationMasksToProtoOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SegmentationMasksToProtoOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field IndexToLabel", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSegmentationMasksToProtoOptions
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
				return ErrInvalidLengthSegmentationMasksToProtoOptions
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthSegmentationMasksToProtoOptions
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.IndexToLabel = append(m.IndexToLabel, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipSegmentationMasksToProtoOptions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthSegmentationMasksToProtoOptions
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
func skipSegmentationMasksToProtoOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowSegmentationMasksToProtoOptions
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
					return 0, ErrIntOverflowSegmentationMasksToProtoOptions
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
					return 0, ErrIntOverflowSegmentationMasksToProtoOptions
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
				return 0, ErrInvalidLengthSegmentationMasksToProtoOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupSegmentationMasksToProtoOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthSegmentationMasksToProtoOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthSegmentationMasksToProtoOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowSegmentationMasksToProtoOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupSegmentationMasksToProtoOptions = fmt.Errorf("proto: unexpected end of group")
)
