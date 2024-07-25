// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/calculators/core/optionspb/classifications_metrics_sink_calculator_options.proto

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

type ClassificationsMetricsSinkCalculatorOptions struct {
	MetricAttributes map[string]string `protobuf:"bytes,1,rep,name=metric_attributes,json=metricAttributes,proto3" json:"metric_attributes,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (m *ClassificationsMetricsSinkCalculatorOptions) Reset() {
	*m = ClassificationsMetricsSinkCalculatorOptions{}
}
func (*ClassificationsMetricsSinkCalculatorOptions) ProtoMessage() {}
func (*ClassificationsMetricsSinkCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_022ad64adbd6d420, []int{0}
}
func (m *ClassificationsMetricsSinkCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ClassificationsMetricsSinkCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ClassificationsMetricsSinkCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ClassificationsMetricsSinkCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ClassificationsMetricsSinkCalculatorOptions.Merge(m, src)
}
func (m *ClassificationsMetricsSinkCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *ClassificationsMetricsSinkCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_ClassificationsMetricsSinkCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_ClassificationsMetricsSinkCalculatorOptions proto.InternalMessageInfo

func (m *ClassificationsMetricsSinkCalculatorOptions) GetMetricAttributes() map[string]string {
	if m != nil {
		return m.MetricAttributes
	}
	return nil
}

func init() {
	proto.RegisterType((*ClassificationsMetricsSinkCalculatorOptions)(nil), "gml.gem.calculators.core.optionspb.ClassificationsMetricsSinkCalculatorOptions")
	proto.RegisterMapType((map[string]string)(nil), "gml.gem.calculators.core.optionspb.ClassificationsMetricsSinkCalculatorOptions.MetricAttributesEntry")
}

func init() {
	proto.RegisterFile("src/gem/calculators/core/optionspb/classifications_metrics_sink_calculator_options.proto", fileDescriptor_022ad64adbd6d420)
}

var fileDescriptor_022ad64adbd6d420 = []byte{
	// 311 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x91, 0xbf, 0x4b, 0x03, 0x31,
	0x14, 0xc7, 0xef, 0xb5, 0x28, 0x78, 0x2e, 0xf5, 0x50, 0x28, 0x0e, 0x8f, 0xd2, 0xa9, 0x20, 0x24,
	0xa0, 0x8b, 0xe8, 0x54, 0x8b, 0xa3, 0x08, 0x75, 0x11, 0x97, 0x23, 0x17, 0xe2, 0x11, 0x9a, 0xbb,
	0x94, 0x24, 0x55, 0xba, 0xb9, 0xb9, 0x8a, 0x7f, 0x85, 0x7f, 0x8a, 0x63, 0xc7, 0x8e, 0x36, 0x5d,
	0x1c, 0xfb, 0x27, 0x48, 0x1b, 0x39, 0x7f, 0x20, 0x14, 0xb7, 0xf7, 0xbe, 0xc9, 0xf7, 0xf3, 0x19,
	0x5e, 0x7c, 0x6d, 0x0d, 0xa7, 0xb9, 0x28, 0x28, 0x67, 0x8a, 0x8f, 0x14, 0x73, 0xda, 0x58, 0xca,
	0xb5, 0x11, 0x54, 0x0f, 0x9d, 0xd4, 0xa5, 0x1d, 0x66, 0x94, 0x2b, 0x66, 0xad, 0xbc, 0x95, 0x9c,
	0xad, 0x92, 0xb4, 0x10, 0xce, 0x48, 0x6e, 0x53, 0x2b, 0xcb, 0x41, 0xfa, 0xd5, 0x4b, 0x3f, 0x1b,
	0x64, 0x68, 0xb4, 0xd3, 0x49, 0x3b, 0x2f, 0x14, 0xc9, 0x45, 0x41, 0xbe, 0x91, 0xc9, 0x92, 0x4c,
	0x2a, 0x72, 0xfb, 0xb1, 0x16, 0x1f, 0xf4, 0x7e, 0xd2, 0x2f, 0x02, 0xfc, 0x4a, 0x96, 0x83, 0x5e,
	0x55, 0xbc, 0x0c, 0x8d, 0xe4, 0x19, 0xe2, 0x9d, 0x60, 0x4f, 0x99, 0x73, 0x46, 0x66, 0x23, 0x27,
	0x6c, 0x13, 0x5a, 0xf5, 0xce, 0xf6, 0xa1, 0x20, 0xeb, 0x85, 0xe4, 0x1f, 0x32, 0x12, 0x1e, 0xbb,
	0x95, 0xe7, 0xbc, 0x74, 0x66, 0xdc, 0x6f, 0x14, 0xbf, 0xe2, 0xfd, 0x5e, 0xbc, 0xf7, 0xe7, 0xd7,
	0xa4, 0x11, 0xd7, 0x07, 0x62, 0xdc, 0x84, 0x16, 0x74, 0xb6, 0xfa, 0xcb, 0x31, 0xd9, 0x8d, 0x37,
	0xee, 0x98, 0x1a, 0x89, 0x66, 0x6d, 0x95, 0x85, 0xe5, 0xa4, 0x76, 0x0c, 0x67, 0xf7, 0x93, 0x19,
	0x46, 0xd3, 0x19, 0x46, 0x8b, 0x19, 0xc2, 0x83, 0x47, 0x78, 0xf1, 0x08, 0xaf, 0x1e, 0x61, 0xe2,
	0x11, 0xde, 0x3c, 0xc2, 0xbb, 0xc7, 0x68, 0xe1, 0x11, 0x9e, 0xe6, 0x18, 0x4d, 0xe6, 0x18, 0x4d,
	0xe7, 0x18, 0xdd, 0x74, 0x73, 0x59, 0x28, 0xe1, 0x14, 0xcb, 0x2c, 0x61, 0x92, 0x86, 0x8d, 0xae,
	0x3f, 0xe9, 0x69, 0x35, 0x65, 0x9b, 0xab, 0x6b, 0x1d, 0x7d, 0x04, 0x00, 0x00, 0xff, 0xff, 0xf7,
	0x36, 0x21, 0x45, 0x09, 0x02, 0x00, 0x00,
}

func (this *ClassificationsMetricsSinkCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ClassificationsMetricsSinkCalculatorOptions)
	if !ok {
		that2, ok := that.(ClassificationsMetricsSinkCalculatorOptions)
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
func (this *ClassificationsMetricsSinkCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&optionspb.ClassificationsMetricsSinkCalculatorOptions{")
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
func valueToGoStringClassificationsMetricsSinkCalculatorOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *ClassificationsMetricsSinkCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ClassificationsMetricsSinkCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ClassificationsMetricsSinkCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
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
			i = encodeVarintClassificationsMetricsSinkCalculatorOptions(dAtA, i, uint64(len(v)))
			i--
			dAtA[i] = 0x12
			i -= len(k)
			copy(dAtA[i:], k)
			i = encodeVarintClassificationsMetricsSinkCalculatorOptions(dAtA, i, uint64(len(k)))
			i--
			dAtA[i] = 0xa
			i = encodeVarintClassificationsMetricsSinkCalculatorOptions(dAtA, i, uint64(baseI-i))
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintClassificationsMetricsSinkCalculatorOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovClassificationsMetricsSinkCalculatorOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ClassificationsMetricsSinkCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.MetricAttributes) > 0 {
		for k, v := range m.MetricAttributes {
			_ = k
			_ = v
			mapEntrySize := 1 + len(k) + sovClassificationsMetricsSinkCalculatorOptions(uint64(len(k))) + 1 + len(v) + sovClassificationsMetricsSinkCalculatorOptions(uint64(len(v)))
			n += mapEntrySize + 1 + sovClassificationsMetricsSinkCalculatorOptions(uint64(mapEntrySize))
		}
	}
	return n
}

func sovClassificationsMetricsSinkCalculatorOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozClassificationsMetricsSinkCalculatorOptions(x uint64) (n int) {
	return sovClassificationsMetricsSinkCalculatorOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ClassificationsMetricsSinkCalculatorOptions) String() string {
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
	s := strings.Join([]string{`&ClassificationsMetricsSinkCalculatorOptions{`,
		`MetricAttributes:` + mapStringForMetricAttributes + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringClassificationsMetricsSinkCalculatorOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ClassificationsMetricsSinkCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
			return fmt.Errorf("proto: ClassificationsMetricsSinkCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ClassificationsMetricsSinkCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field MetricAttributes", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
				return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
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
						return ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
							return ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
						return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
					}
					postStringIndexmapkey := iNdEx + intStringLenmapkey
					if postStringIndexmapkey < 0 {
						return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
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
							return ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
						return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
					}
					postStringIndexmapvalue := iNdEx + intStringLenmapvalue
					if postStringIndexmapvalue < 0 {
						return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
					}
					if postStringIndexmapvalue > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = string(dAtA[iNdEx:postStringIndexmapvalue])
					iNdEx = postStringIndexmapvalue
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipClassificationsMetricsSinkCalculatorOptions(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if (skippy < 0) || (iNdEx+skippy) < 0 {
						return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
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
			skippy, err := skipClassificationsMetricsSinkCalculatorOptions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
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
func skipClassificationsMetricsSinkCalculatorOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowClassificationsMetricsSinkCalculatorOptions
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
				return 0, ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupClassificationsMetricsSinkCalculatorOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthClassificationsMetricsSinkCalculatorOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowClassificationsMetricsSinkCalculatorOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupClassificationsMetricsSinkCalculatorOptions = fmt.Errorf("proto: unexpected end of group")
)
