// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/calculators/core/optionspb/semantic_segmentation_metrics_sink_calculator_options.proto

package optionspb

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	github_com_gogo_protobuf_sortkeys "github.com/gogo/protobuf/sortkeys"
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

type SemanticSegmentationMetricsSinkCalculatorOptions struct {
	MetricAttributes map[string]string `protobuf:"bytes,1,rep,name=metric_attributes,json=metricAttributes,proto3" json:"metric_attributes,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (m *SemanticSegmentationMetricsSinkCalculatorOptions) Reset() {
	*m = SemanticSegmentationMetricsSinkCalculatorOptions{}
}
func (*SemanticSegmentationMetricsSinkCalculatorOptions) ProtoMessage() {}
func (*SemanticSegmentationMetricsSinkCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_41774f65b6df2711, []int{0}
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_SemanticSegmentationMetricsSinkCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SemanticSegmentationMetricsSinkCalculatorOptions.Merge(m, src)
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_SemanticSegmentationMetricsSinkCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_SemanticSegmentationMetricsSinkCalculatorOptions proto.InternalMessageInfo

func (m *SemanticSegmentationMetricsSinkCalculatorOptions) GetMetricAttributes() map[string]string {
	if m != nil {
		return m.MetricAttributes
	}
	return nil
}

func init() {
	proto.RegisterType((*SemanticSegmentationMetricsSinkCalculatorOptions)(nil), "gml.gem.calculators.core.optionspb.SemanticSegmentationMetricsSinkCalculatorOptions")
	proto.RegisterMapType((map[string]string)(nil), "gml.gem.calculators.core.optionspb.SemanticSegmentationMetricsSinkCalculatorOptions.MetricAttributesEntry")
}

func init() {
	proto.RegisterFile("src/gem/calculators/core/optionspb/semantic_segmentation_metrics_sink_calculator_options.proto", fileDescriptor_41774f65b6df2711)
}

var fileDescriptor_41774f65b6df2711 = []byte{
	// 318 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x91, 0x3f, 0x4b, 0x03, 0x31,
	0x18, 0x87, 0xef, 0x6d, 0x51, 0xf0, 0x5c, 0xea, 0xa1, 0x50, 0x1c, 0x5e, 0x4a, 0xa7, 0x4e, 0x89,
	0xe8, 0x22, 0x3a, 0xd5, 0xe2, 0x28, 0x42, 0xbb, 0x39, 0x78, 0xe4, 0x8e, 0x70, 0xc4, 0x5e, 0x2e,
	0x25, 0x49, 0x95, 0x6e, 0x7e, 0x04, 0x9d, 0xfc, 0x0a, 0x7e, 0x14, 0xc7, 0x8e, 0x1d, 0x6d, 0xba,
	0x38, 0xf6, 0x23, 0xc8, 0x35, 0xe5, 0x2c, 0x22, 0x14, 0xb7, 0xf7, 0x4f, 0x9e, 0xdf, 0x13, 0x78,
	0xc3, 0x7b, 0xa3, 0x53, 0x9a, 0x71, 0x49, 0x53, 0x96, 0xa7, 0xe3, 0x9c, 0x59, 0xa5, 0x0d, 0x4d,
	0x95, 0xe6, 0x54, 0x8d, 0xac, 0x50, 0x85, 0x19, 0x25, 0xd4, 0x70, 0xc9, 0x0a, 0x2b, 0xd2, 0xd8,
	0xf0, 0x4c, 0xf2, 0xc2, 0xb2, 0x72, 0x11, 0x4b, 0x6e, 0xb5, 0x48, 0x4d, 0x6c, 0x44, 0x31, 0x8c,
	0x7f, 0xe8, 0x78, 0xcd, 0x91, 0x91, 0x56, 0x56, 0x45, 0xed, 0x4c, 0xe6, 0x24, 0xe3, 0x92, 0x6c,
	0xe4, 0x93, 0x32, 0x9f, 0x54, 0xf9, 0xed, 0xd7, 0x5a, 0x78, 0x32, 0x58, 0x3b, 0x06, 0x1b, 0x8a,
	0x1b, 0x6f, 0x18, 0x88, 0x62, 0xd8, 0xab, 0xe8, 0x5b, 0x8f, 0x45, 0x6f, 0x10, 0x1e, 0xf8, 0x2f,
	0xc4, 0xcc, 0x5a, 0x2d, 0x92, 0xb1, 0xe5, 0xa6, 0x09, 0xad, 0x7a, 0x67, 0xff, 0xf4, 0x81, 0x6c,
	0xb7, 0x92, 0xff, 0x1a, 0x89, 0x5f, 0x76, 0x2b, 0xd9, 0x75, 0x61, 0xf5, 0xa4, 0xdf, 0x90, 0xbf,
	0xc6, 0xc7, 0xbd, 0xf0, 0xe8, 0xcf, 0xa7, 0x51, 0x23, 0xac, 0x0f, 0xf9, 0xa4, 0x09, 0x2d, 0xe8,
	0xec, 0xf5, 0xcb, 0x32, 0x3a, 0x0c, 0x77, 0x1e, 0x59, 0x3e, 0xe6, 0xcd, 0xda, 0x6a, 0xe6, 0x9b,
	0x8b, 0xda, 0x39, 0x5c, 0x3d, 0x4d, 0xe7, 0x18, 0xcc, 0xe6, 0x18, 0x2c, 0xe7, 0x08, 0xcf, 0x0e,
	0xe1, 0xdd, 0x21, 0x7c, 0x38, 0x84, 0xa9, 0x43, 0xf8, 0x74, 0x08, 0x5f, 0x0e, 0x83, 0xa5, 0x43,
	0x78, 0x59, 0x60, 0x30, 0x5d, 0x60, 0x30, 0x5b, 0x60, 0x70, 0xd7, 0xcd, 0x84, 0xcc, 0xb9, 0xcd,
	0x59, 0x62, 0x08, 0x13, 0xd4, 0x77, 0x74, 0xfb, 0x89, 0x2f, 0xab, 0x2a, 0xd9, 0x5d, 0xdd, 0xed,
	0xec, 0x3b, 0x00, 0x00, 0xff, 0xff, 0x2e, 0x23, 0xfd, 0xe3, 0x19, 0x02, 0x00, 0x00,
}

func (this *SemanticSegmentationMetricsSinkCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*SemanticSegmentationMetricsSinkCalculatorOptions)
	if !ok {
		that2, ok := that.(SemanticSegmentationMetricsSinkCalculatorOptions)
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
	if len(this.MetricAttributes) != len(that1.MetricAttributes) {
		return false
	}
	for i := range this.MetricAttributes {
		if this.MetricAttributes[i] != that1.MetricAttributes[i] {
			return false
		}
	}
	return true
}
func (this *SemanticSegmentationMetricsSinkCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&optionspb.SemanticSegmentationMetricsSinkCalculatorOptions{")
	keysForMetricAttributes := make([]string, 0, len(this.MetricAttributes))
	for k, _ := range this.MetricAttributes {
		keysForMetricAttributes = append(keysForMetricAttributes, k)
	}
	github_com_gogo_protobuf_sortkeys.Strings(keysForMetricAttributes)
	mapStringForMetricAttributes := "map[string]string{"
	for _, k := range keysForMetricAttributes {
		mapStringForMetricAttributes += fmt.Sprintf("%#v: %#v,", k, this.MetricAttributes[k])
	}
	mapStringForMetricAttributes += "}"
	if this.MetricAttributes != nil {
		s = append(s, "MetricAttributes: "+mapStringForMetricAttributes+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringSemanticSegmentationMetricsSinkCalculatorOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SemanticSegmentationMetricsSinkCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SemanticSegmentationMetricsSinkCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.MetricAttributes) > 0 {
		for k := range m.MetricAttributes {
			v := m.MetricAttributes[k]
			baseI := i
			i -= len(v)
			copy(dAtA[i:], v)
			i = encodeVarintSemanticSegmentationMetricsSinkCalculatorOptions(dAtA, i, uint64(len(v)))
			i--
			dAtA[i] = 0x12
			i -= len(k)
			copy(dAtA[i:], k)
			i = encodeVarintSemanticSegmentationMetricsSinkCalculatorOptions(dAtA, i, uint64(len(k)))
			i--
			dAtA[i] = 0xa
			i = encodeVarintSemanticSegmentationMetricsSinkCalculatorOptions(dAtA, i, uint64(baseI-i))
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintSemanticSegmentationMetricsSinkCalculatorOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovSemanticSegmentationMetricsSinkCalculatorOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.MetricAttributes) > 0 {
		for k, v := range m.MetricAttributes {
			_ = k
			_ = v
			mapEntrySize := 1 + len(k) + sovSemanticSegmentationMetricsSinkCalculatorOptions(uint64(len(k))) + 1 + len(v) + sovSemanticSegmentationMetricsSinkCalculatorOptions(uint64(len(v)))
			n += mapEntrySize + 1 + sovSemanticSegmentationMetricsSinkCalculatorOptions(uint64(mapEntrySize))
		}
	}
	return n
}

func sovSemanticSegmentationMetricsSinkCalculatorOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozSemanticSegmentationMetricsSinkCalculatorOptions(x uint64) (n int) {
	return sovSemanticSegmentationMetricsSinkCalculatorOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *SemanticSegmentationMetricsSinkCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	keysForMetricAttributes := make([]string, 0, len(this.MetricAttributes))
	for k, _ := range this.MetricAttributes {
		keysForMetricAttributes = append(keysForMetricAttributes, k)
	}
	github_com_gogo_protobuf_sortkeys.Strings(keysForMetricAttributes)
	mapStringForMetricAttributes := "map[string]string{"
	for _, k := range keysForMetricAttributes {
		mapStringForMetricAttributes += fmt.Sprintf("%v: %v,", k, this.MetricAttributes[k])
	}
	mapStringForMetricAttributes += "}"
	s := strings.Join([]string{`&SemanticSegmentationMetricsSinkCalculatorOptions{`,
		`MetricAttributes:` + mapStringForMetricAttributes + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringSemanticSegmentationMetricsSinkCalculatorOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *SemanticSegmentationMetricsSinkCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
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
			return fmt.Errorf("proto: SemanticSegmentationMetricsSinkCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SemanticSegmentationMetricsSinkCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field MetricAttributes", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
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
				return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.MetricAttributes == nil {
				m.MetricAttributes = make(map[string]string)
			}
			var mapkey string
			var mapvalue string
			for iNdEx < postIndex {
				entryPreIndex := iNdEx
				var wire uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
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
				if fieldNum == 1 {
					var stringLenmapkey uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						stringLenmapkey |= uint64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					intStringLenmapkey := int(stringLenmapkey)
					if intStringLenmapkey < 0 {
						return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
					}
					postStringIndexmapkey := iNdEx + intStringLenmapkey
					if postStringIndexmapkey < 0 {
						return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
					}
					if postStringIndexmapkey > l {
						return io.ErrUnexpectedEOF
					}
					mapkey = string(dAtA[iNdEx:postStringIndexmapkey])
					iNdEx = postStringIndexmapkey
				} else if fieldNum == 2 {
					var stringLenmapvalue uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						stringLenmapvalue |= uint64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					intStringLenmapvalue := int(stringLenmapvalue)
					if intStringLenmapvalue < 0 {
						return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
					}
					postStringIndexmapvalue := iNdEx + intStringLenmapvalue
					if postStringIndexmapvalue < 0 {
						return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
					}
					if postStringIndexmapvalue > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = string(dAtA[iNdEx:postStringIndexmapvalue])
					iNdEx = postStringIndexmapvalue
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipSemanticSegmentationMetricsSinkCalculatorOptions(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if (skippy < 0) || (iNdEx+skippy) < 0 {
						return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
					}
					if (iNdEx + skippy) > postIndex {
						return io.ErrUnexpectedEOF
					}
					iNdEx += skippy
				}
			}
			m.MetricAttributes[mapkey] = mapvalue
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipSemanticSegmentationMetricsSinkCalculatorOptions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
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
func skipSemanticSegmentationMetricsSinkCalculatorOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions
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
				return 0, ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupSemanticSegmentationMetricsSinkCalculatorOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthSemanticSegmentationMetricsSinkCalculatorOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowSemanticSegmentationMetricsSinkCalculatorOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupSemanticSegmentationMetricsSinkCalculatorOptions = fmt.Errorf("proto: unexpected end of group")
)
