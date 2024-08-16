// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/calculators/core/optionspb/flow_limiter_metrics_sink_calculator_options.proto

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

type FlowLimiterMetricsSinkCalculatorOptions struct {
	MetricAttributes map[string]string `protobuf:"bytes,1,rep,name=metric_attributes,json=metricAttributes,proto3" json:"metric_attributes,omitempty" protobuf_key:"bytes,1,opt,name=key,proto3" protobuf_val:"bytes,2,opt,name=value,proto3"`
}

func (m *FlowLimiterMetricsSinkCalculatorOptions) Reset() {
	*m = FlowLimiterMetricsSinkCalculatorOptions{}
}
func (*FlowLimiterMetricsSinkCalculatorOptions) ProtoMessage() {}
func (*FlowLimiterMetricsSinkCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_93980906d6c21a1d, []int{0}
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_FlowLimiterMetricsSinkCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FlowLimiterMetricsSinkCalculatorOptions.Merge(m, src)
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_FlowLimiterMetricsSinkCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_FlowLimiterMetricsSinkCalculatorOptions proto.InternalMessageInfo

func (m *FlowLimiterMetricsSinkCalculatorOptions) GetMetricAttributes() map[string]string {
	if m != nil {
		return m.MetricAttributes
	}
	return nil
}

func init() {
	proto.RegisterType((*FlowLimiterMetricsSinkCalculatorOptions)(nil), "gml.gem.calculators.core.optionspb.FlowLimiterMetricsSinkCalculatorOptions")
	proto.RegisterMapType((map[string]string)(nil), "gml.gem.calculators.core.optionspb.FlowLimiterMetricsSinkCalculatorOptions.MetricAttributesEntry")
}

func init() {
	proto.RegisterFile("src/gem/calculators/core/optionspb/flow_limiter_metrics_sink_calculator_options.proto", fileDescriptor_93980906d6c21a1d)
}

var fileDescriptor_93980906d6c21a1d = []byte{
	// 315 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x91, 0xbf, 0x4a, 0x2b, 0x41,
	0x14, 0x87, 0xf7, 0x24, 0xdc, 0x0b, 0x77, 0x6f, 0x13, 0x17, 0x85, 0x60, 0x71, 0x08, 0x69, 0x4c,
	0x35, 0x03, 0xda, 0x88, 0x56, 0x31, 0x68, 0xa5, 0x08, 0x11, 0x1b, 0x9b, 0x65, 0x76, 0x19, 0x97,
	0x21, 0x33, 0x3b, 0x61, 0x66, 0x62, 0x48, 0xe7, 0x13, 0x88, 0x8f, 0xe1, 0xa3, 0x58, 0xa6, 0x4c,
	0x69, 0x26, 0x8d, 0x65, 0xde, 0x40, 0x49, 0x46, 0x56, 0x11, 0x21, 0x76, 0xe7, 0xcf, 0xef, 0x7c,
	0x5f, 0x71, 0xe2, 0x6b, 0x6b, 0x72, 0x5a, 0x70, 0x45, 0x73, 0x26, 0xf3, 0x91, 0x64, 0x4e, 0x1b,
	0x4b, 0x73, 0x6d, 0x38, 0xd5, 0x43, 0x27, 0x74, 0x69, 0x87, 0x19, 0xbd, 0x95, 0x7a, 0x9c, 0x4a,
	0xa1, 0x84, 0xe3, 0x26, 0x55, 0xdc, 0x19, 0x91, 0xdb, 0xd4, 0x8a, 0x72, 0x90, 0x7e, 0x1e, 0xa5,
	0x1f, 0x71, 0x32, 0x34, 0xda, 0xe9, 0xa4, 0x5d, 0x28, 0x49, 0x0a, 0xae, 0xc8, 0x17, 0x2c, 0x59,
	0x61, 0x49, 0x85, 0x6d, 0xbf, 0x41, 0xbc, 0x77, 0x26, 0xf5, 0xf8, 0x3c, 0x90, 0x2f, 0x02, 0xf8,
	0x4a, 0x94, 0x83, 0x5e, 0x75, 0x74, 0x19, 0xd2, 0xc9, 0x03, 0xc4, 0x5b, 0xc1, 0x9c, 0x32, 0xe7,
	0x8c, 0xc8, 0x46, 0x8e, 0xdb, 0x26, 0xb4, 0xea, 0x9d, 0xff, 0xfb, 0x8c, 0x6c, 0x96, 0x91, 0x5f,
	0x8a, 0x48, 0x58, 0x76, 0x2b, 0xc7, 0x69, 0xe9, 0xcc, 0xa4, 0xdf, 0x50, 0xdf, 0xc6, 0xbb, 0xbd,
	0x78, 0xe7, 0xc7, 0x68, 0xd2, 0x88, 0xeb, 0x03, 0x3e, 0x69, 0x42, 0x0b, 0x3a, 0xff, 0xfa, 0xab,
	0x32, 0xd9, 0x8e, 0xff, 0xdc, 0x31, 0x39, 0xe2, 0xcd, 0xda, 0x7a, 0x16, 0x9a, 0xa3, 0xda, 0x21,
	0x9c, 0x8c, 0xa7, 0x73, 0x8c, 0x66, 0x73, 0x8c, 0x96, 0x73, 0x84, 0x7b, 0x8f, 0xf0, 0xe4, 0x11,
	0x9e, 0x3d, 0xc2, 0xd4, 0x23, 0xbc, 0x78, 0x84, 0x57, 0x8f, 0xd1, 0xd2, 0x23, 0x3c, 0x2e, 0x30,
	0x9a, 0x2e, 0x30, 0x9a, 0x2d, 0x30, 0xba, 0xe9, 0x16, 0x42, 0x49, 0xee, 0x24, 0xcb, 0x2c, 0x61,
	0x82, 0x86, 0x8e, 0x6e, 0xfe, 0xe3, 0x71, 0x55, 0x65, 0x7f, 0xd7, 0x5f, 0x3a, 0x78, 0x0f, 0x00,
	0x00, 0xff, 0xff, 0x4d, 0xdb, 0x50, 0xe7, 0xfe, 0x01, 0x00, 0x00,
}

func (this *FlowLimiterMetricsSinkCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*FlowLimiterMetricsSinkCalculatorOptions)
	if !ok {
		that2, ok := that.(FlowLimiterMetricsSinkCalculatorOptions)
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
func (this *FlowLimiterMetricsSinkCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&optionspb.FlowLimiterMetricsSinkCalculatorOptions{")
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
func valueToGoStringFlowLimiterMetricsSinkCalculatorOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *FlowLimiterMetricsSinkCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *FlowLimiterMetricsSinkCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
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
			i = encodeVarintFlowLimiterMetricsSinkCalculatorOptions(dAtA, i, uint64(len(v)))
			i--
			dAtA[i] = 0x12
			i -= len(k)
			copy(dAtA[i:], k)
			i = encodeVarintFlowLimiterMetricsSinkCalculatorOptions(dAtA, i, uint64(len(k)))
			i--
			dAtA[i] = 0xa
			i = encodeVarintFlowLimiterMetricsSinkCalculatorOptions(dAtA, i, uint64(baseI-i))
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintFlowLimiterMetricsSinkCalculatorOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovFlowLimiterMetricsSinkCalculatorOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.MetricAttributes) > 0 {
		for k, v := range m.MetricAttributes {
			_ = k
			_ = v
			mapEntrySize := 1 + len(k) + sovFlowLimiterMetricsSinkCalculatorOptions(uint64(len(k))) + 1 + len(v) + sovFlowLimiterMetricsSinkCalculatorOptions(uint64(len(v)))
			n += mapEntrySize + 1 + sovFlowLimiterMetricsSinkCalculatorOptions(uint64(mapEntrySize))
		}
	}
	return n
}

func sovFlowLimiterMetricsSinkCalculatorOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozFlowLimiterMetricsSinkCalculatorOptions(x uint64) (n int) {
	return sovFlowLimiterMetricsSinkCalculatorOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *FlowLimiterMetricsSinkCalculatorOptions) String() string {
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
	s := strings.Join([]string{`&FlowLimiterMetricsSinkCalculatorOptions{`,
		`MetricAttributes:` + mapStringForMetricAttributes + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringFlowLimiterMetricsSinkCalculatorOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *FlowLimiterMetricsSinkCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
			return fmt.Errorf("proto: FlowLimiterMetricsSinkCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: FlowLimiterMetricsSinkCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field MetricAttributes", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
				return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
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
						return ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
							return ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
						return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
					}
					postStringIndexmapkey := iNdEx + intStringLenmapkey
					if postStringIndexmapkey < 0 {
						return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
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
							return ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
						return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
					}
					postStringIndexmapvalue := iNdEx + intStringLenmapvalue
					if postStringIndexmapvalue < 0 {
						return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
					}
					if postStringIndexmapvalue > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = string(dAtA[iNdEx:postStringIndexmapvalue])
					iNdEx = postStringIndexmapvalue
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipFlowLimiterMetricsSinkCalculatorOptions(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if (skippy < 0) || (iNdEx+skippy) < 0 {
						return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
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
			skippy, err := skipFlowLimiterMetricsSinkCalculatorOptions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
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
func skipFlowLimiterMetricsSinkCalculatorOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
					return 0, ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions
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
				return 0, ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupFlowLimiterMetricsSinkCalculatorOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthFlowLimiterMetricsSinkCalculatorOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowFlowLimiterMetricsSinkCalculatorOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupFlowLimiterMetricsSinkCalculatorOptions = fmt.Errorf("proto: unexpected end of group")
)
