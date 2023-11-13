// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/tool/testdata/frozen_generator.proto

package testdata

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	github_com_gogo_protobuf_sortkeys "github.com/gogo/protobuf/sortkeys"
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

type FrozenGeneratorOptions struct {
	GraphProtoPath        string            `protobuf:"bytes,1,opt,name=graph_proto_path,json=graphProtoPath" json:"graph_proto_path"`
	TagToTensorNames      map[string]string `protobuf:"bytes,2,rep,name=tag_to_tensor_names,json=tagToTensorNames" json:"tag_to_tensor_names,omitempty" protobuf_key:"bytes,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
	InitializationOpNames []string          `protobuf:"bytes,4,rep,name=initialization_op_names,json=initializationOpNames" json:"initialization_op_names,omitempty"`
}

func (m *FrozenGeneratorOptions) Reset()      { *m = FrozenGeneratorOptions{} }
func (*FrozenGeneratorOptions) ProtoMessage() {}
func (*FrozenGeneratorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_6415082bdaac052b, []int{0}
}
func (m *FrozenGeneratorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *FrozenGeneratorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_FrozenGeneratorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *FrozenGeneratorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FrozenGeneratorOptions.Merge(m, src)
}
func (m *FrozenGeneratorOptions) XXX_Size() int {
	return m.Size()
}
func (m *FrozenGeneratorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_FrozenGeneratorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_FrozenGeneratorOptions proto.InternalMessageInfo

func (m *FrozenGeneratorOptions) GetGraphProtoPath() string {
	if m != nil {
		return m.GraphProtoPath
	}
	return ""
}

func (m *FrozenGeneratorOptions) GetTagToTensorNames() map[string]string {
	if m != nil {
		return m.TagToTensorNames
	}
	return nil
}

func (m *FrozenGeneratorOptions) GetInitializationOpNames() []string {
	if m != nil {
		return m.InitializationOpNames
	}
	return nil
}

var E_FrozenGeneratorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.PacketGeneratorOptions)(nil),
	ExtensionType: (*FrozenGeneratorOptions)(nil),
	Field:         225748738,
	Name:          "mediapipe.FrozenGeneratorOptions.ext",
	Tag:           "bytes,225748738,opt,name=ext",
	Filename:      "mediapipe/framework/tool/testdata/frozen_generator.proto",
}

func init() {
	proto.RegisterExtension(E_FrozenGeneratorOptions_Ext)
	proto.RegisterType((*FrozenGeneratorOptions)(nil), "mediapipe.FrozenGeneratorOptions")
	proto.RegisterMapType((map[string]string)(nil), "mediapipe.FrozenGeneratorOptions.TagToTensorNamesEntry")
}

func init() {
	proto.RegisterFile("mediapipe/framework/tool/testdata/frozen_generator.proto", fileDescriptor_6415082bdaac052b)
}

var fileDescriptor_6415082bdaac052b = []byte{
	// 394 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x92, 0xbd, 0x8e, 0xda, 0x40,
	0x14, 0x85, 0x3d, 0x98, 0x14, 0x0c, 0x52, 0x84, 0x1c, 0x41, 0x2c, 0x8a, 0x89, 0x49, 0x65, 0xa5,
	0xb0, 0x25, 0x8a, 0x04, 0x45, 0x4a, 0x83, 0x94, 0xa4, 0x0b, 0xc8, 0xa2, 0x49, 0x1a, 0x6b, 0x02,
	0x83, 0x3d, 0xf2, 0xcf, 0x8c, 0xc6, 0x97, 0x24, 0x50, 0x45, 0x79, 0x82, 0x14, 0x79, 0x88, 0x3c,
	0x0a, 0x55, 0x84, 0x52, 0x51, 0xad, 0x16, 0xd3, 0x6c, 0xc9, 0x23, 0xac, 0xec, 0xdd, 0x65, 0x01,
	0x21, 0xd1, 0xde, 0x33, 0xdf, 0x39, 0x73, 0xae, 0x2e, 0xee, 0x25, 0x6c, 0xc2, 0xa9, 0xe4, 0x92,
	0xb9, 0x53, 0x45, 0x13, 0xf6, 0x5d, 0xa8, 0xc8, 0x05, 0x21, 0x62, 0x17, 0x58, 0x06, 0x13, 0x0a,
	0xd4, 0x9d, 0x2a, 0xb1, 0x60, 0xa9, 0x1f, 0xb0, 0x94, 0x29, 0x0a, 0x42, 0x39, 0x52, 0x09, 0x10,
	0x46, 0x6d, 0x4f, 0xb6, 0x5f, 0x9d, 0x33, 0x91, 0x74, 0x1c, 0x31, 0x38, 0xc5, 0x5e, 0xfe, 0xd1,
	0x71, 0xeb, 0x43, 0xe9, 0xf8, 0xf1, 0x41, 0x19, 0x48, 0xe0, 0x22, 0xcd, 0x0c, 0x07, 0x37, 0x02,
	0x45, 0x65, 0xe8, 0x97, 0x2f, 0x7d, 0x49, 0x21, 0x34, 0x91, 0x85, 0xec, 0x5a, 0xbf, 0xba, 0xbc,
	0x7a, 0xa1, 0x79, 0x4f, 0x4b, 0x75, 0x58, 0x88, 0x43, 0x0a, 0xa1, 0x31, 0xc5, 0xcf, 0x80, 0x06,
	0x3e, 0x08, 0x1f, 0x58, 0x9a, 0x09, 0xe5, 0xa7, 0x34, 0x61, 0x99, 0x59, 0xb1, 0x74, 0xbb, 0xde,
	0x7d, 0xe3, 0xec, 0x3f, 0xe5, 0x9c, 0xcf, 0x73, 0x46, 0x34, 0x18, 0x89, 0x51, 0x89, 0x7e, 0x2a,
	0xc8, 0xf7, 0x29, 0xa8, 0xb9, 0xd7, 0x80, 0x93, 0xb1, 0xf1, 0x1a, 0x3f, 0xe7, 0x29, 0x07, 0x4e,
	0x63, 0xbe, 0xa0, 0x05, 0xea, 0x0b, 0x79, 0x9f, 0x55, 0xb5, 0x74, 0xbb, 0xe6, 0x35, 0x8f, 0xe5,
	0x81, 0x2c, 0xb9, 0xf6, 0x00, 0x37, 0xcf, 0x46, 0x18, 0x2d, 0xac, 0x47, 0x6c, 0x7e, 0xd4, 0xad,
	0x18, 0x18, 0x6d, 0xfc, 0xe4, 0x1b, 0x8d, 0x67, 0xcc, 0xac, 0x1c, 0x28, 0x77, 0xa3, 0xb7, 0x95,
	0x1e, 0xea, 0x7e, 0xc6, 0x3a, 0xfb, 0x01, 0x46, 0xe7, 0xa0, 0xda, 0xb0, 0xdc, 0xf2, 0x69, 0x35,
	0xf3, 0xd7, 0xbf, 0xff, 0x91, 0x85, 0xec, 0x7a, 0xb7, 0x73, 0x71, 0x0d, 0x5e, 0xe1, 0xd9, 0xcf,
	0x56, 0x1b, 0xa2, 0xad, 0x37, 0x44, 0xdb, 0x6d, 0x08, 0xfa, 0x99, 0x13, 0xf4, 0x37, 0x27, 0x68,
	0x99, 0x13, 0xb4, 0xca, 0x09, 0xba, 0xce, 0x09, 0xba, 0xc9, 0x89, 0xb6, 0xcb, 0x09, 0xfa, 0xbd,
	0x25, 0xda, 0x6a, 0x4b, 0xb4, 0xf5, 0x96, 0x68, 0x5f, 0xde, 0x05, 0x1c, 0xc2, 0xd9, 0x57, 0x67,
	0x2c, 0x12, 0x37, 0x10, 0x22, 0x88, 0x99, 0xfb, 0x78, 0x0a, 0x17, 0x2f, 0xeb, 0x36, 0x00, 0x00,
	0xff, 0xff, 0x00, 0xbc, 0xc9, 0x7a, 0x7d, 0x02, 0x00, 0x00,
}

func (this *FrozenGeneratorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*FrozenGeneratorOptions)
	if !ok {
		that2, ok := that.(FrozenGeneratorOptions)
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
	if this.GraphProtoPath != that1.GraphProtoPath {
		return false
	}
	if len(this.TagToTensorNames) != len(that1.TagToTensorNames) {
		return false
	}
	for i := range this.TagToTensorNames {
		if this.TagToTensorNames[i] != that1.TagToTensorNames[i] {
			return false
		}
	}
	if len(this.InitializationOpNames) != len(that1.InitializationOpNames) {
		return false
	}
	for i := range this.InitializationOpNames {
		if this.InitializationOpNames[i] != that1.InitializationOpNames[i] {
			return false
		}
	}
	return true
}
func (this *FrozenGeneratorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&testdata.FrozenGeneratorOptions{")
	s = append(s, "GraphProtoPath: "+fmt.Sprintf("%#v", this.GraphProtoPath)+",\n")
	keysForTagToTensorNames := make([]string, 0, len(this.TagToTensorNames))
	for k, _ := range this.TagToTensorNames {
		keysForTagToTensorNames = append(keysForTagToTensorNames, k)
	}
	github_com_gogo_protobuf_sortkeys.Strings(keysForTagToTensorNames)
	mapStringForTagToTensorNames := "map[string]string{"
	for _, k := range keysForTagToTensorNames {
		mapStringForTagToTensorNames += fmt.Sprintf("%#v: %#v,", k, this.TagToTensorNames[k])
	}
	mapStringForTagToTensorNames += "}"
	if this.TagToTensorNames != nil {
		s = append(s, "TagToTensorNames: "+mapStringForTagToTensorNames+",\n")
	}
	if this.InitializationOpNames != nil {
		s = append(s, "InitializationOpNames: "+fmt.Sprintf("%#v", this.InitializationOpNames)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringFrozenGenerator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *FrozenGeneratorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *FrozenGeneratorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *FrozenGeneratorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.InitializationOpNames) > 0 {
		for iNdEx := len(m.InitializationOpNames) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.InitializationOpNames[iNdEx])
			copy(dAtA[i:], m.InitializationOpNames[iNdEx])
			i = encodeVarintFrozenGenerator(dAtA, i, uint64(len(m.InitializationOpNames[iNdEx])))
			i--
			dAtA[i] = 0x22
		}
	}
	if len(m.TagToTensorNames) > 0 {
		for k := range m.TagToTensorNames {
			v := m.TagToTensorNames[k]
			baseI := i
			i -= len(v)
			copy(dAtA[i:], v)
			i = encodeVarintFrozenGenerator(dAtA, i, uint64(len(v)))
			i--
			dAtA[i] = 0x12
			i -= len(k)
			copy(dAtA[i:], k)
			i = encodeVarintFrozenGenerator(dAtA, i, uint64(len(k)))
			i--
			dAtA[i] = 0xa
			i = encodeVarintFrozenGenerator(dAtA, i, uint64(baseI-i))
			i--
			dAtA[i] = 0x12
		}
	}
	i -= len(m.GraphProtoPath)
	copy(dAtA[i:], m.GraphProtoPath)
	i = encodeVarintFrozenGenerator(dAtA, i, uint64(len(m.GraphProtoPath)))
	i--
	dAtA[i] = 0xa
	return len(dAtA) - i, nil
}

func encodeVarintFrozenGenerator(dAtA []byte, offset int, v uint64) int {
	offset -= sovFrozenGenerator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *FrozenGeneratorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.GraphProtoPath)
	n += 1 + l + sovFrozenGenerator(uint64(l))
	if len(m.TagToTensorNames) > 0 {
		for k, v := range m.TagToTensorNames {
			_ = k
			_ = v
			mapEntrySize := 1 + len(k) + sovFrozenGenerator(uint64(len(k))) + 1 + len(v) + sovFrozenGenerator(uint64(len(v)))
			n += mapEntrySize + 1 + sovFrozenGenerator(uint64(mapEntrySize))
		}
	}
	if len(m.InitializationOpNames) > 0 {
		for _, s := range m.InitializationOpNames {
			l = len(s)
			n += 1 + l + sovFrozenGenerator(uint64(l))
		}
	}
	return n
}

func sovFrozenGenerator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozFrozenGenerator(x uint64) (n int) {
	return sovFrozenGenerator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *FrozenGeneratorOptions) String() string {
	if this == nil {
		return "nil"
	}
	keysForTagToTensorNames := make([]string, 0, len(this.TagToTensorNames))
	for k, _ := range this.TagToTensorNames {
		keysForTagToTensorNames = append(keysForTagToTensorNames, k)
	}
	github_com_gogo_protobuf_sortkeys.Strings(keysForTagToTensorNames)
	mapStringForTagToTensorNames := "map[string]string{"
	for _, k := range keysForTagToTensorNames {
		mapStringForTagToTensorNames += fmt.Sprintf("%v: %v,", k, this.TagToTensorNames[k])
	}
	mapStringForTagToTensorNames += "}"
	s := strings.Join([]string{`&FrozenGeneratorOptions{`,
		`GraphProtoPath:` + fmt.Sprintf("%v", this.GraphProtoPath) + `,`,
		`TagToTensorNames:` + mapStringForTagToTensorNames + `,`,
		`InitializationOpNames:` + fmt.Sprintf("%v", this.InitializationOpNames) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringFrozenGenerator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *FrozenGeneratorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowFrozenGenerator
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
			return fmt.Errorf("proto: FrozenGeneratorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: FrozenGeneratorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field GraphProtoPath", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowFrozenGenerator
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
				return ErrInvalidLengthFrozenGenerator
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthFrozenGenerator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.GraphProtoPath = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field TagToTensorNames", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowFrozenGenerator
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
				return ErrInvalidLengthFrozenGenerator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthFrozenGenerator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.TagToTensorNames == nil {
				m.TagToTensorNames = make(map[string]string)
			}
			var mapkey string
			var mapvalue string
			for iNdEx < postIndex {
				entryPreIndex := iNdEx
				var wire uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowFrozenGenerator
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
							return ErrIntOverflowFrozenGenerator
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
						return ErrInvalidLengthFrozenGenerator
					}
					postStringIndexmapkey := iNdEx + intStringLenmapkey
					if postStringIndexmapkey < 0 {
						return ErrInvalidLengthFrozenGenerator
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
							return ErrIntOverflowFrozenGenerator
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
						return ErrInvalidLengthFrozenGenerator
					}
					postStringIndexmapvalue := iNdEx + intStringLenmapvalue
					if postStringIndexmapvalue < 0 {
						return ErrInvalidLengthFrozenGenerator
					}
					if postStringIndexmapvalue > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = string(dAtA[iNdEx:postStringIndexmapvalue])
					iNdEx = postStringIndexmapvalue
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipFrozenGenerator(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if (skippy < 0) || (iNdEx+skippy) < 0 {
						return ErrInvalidLengthFrozenGenerator
					}
					if (iNdEx + skippy) > postIndex {
						return io.ErrUnexpectedEOF
					}
					iNdEx += skippy
				}
			}
			m.TagToTensorNames[mapkey] = mapvalue
			iNdEx = postIndex
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field InitializationOpNames", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowFrozenGenerator
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
				return ErrInvalidLengthFrozenGenerator
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthFrozenGenerator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.InitializationOpNames = append(m.InitializationOpNames, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipFrozenGenerator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthFrozenGenerator
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
func skipFrozenGenerator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowFrozenGenerator
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
					return 0, ErrIntOverflowFrozenGenerator
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
					return 0, ErrIntOverflowFrozenGenerator
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
				return 0, ErrInvalidLengthFrozenGenerator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupFrozenGenerator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthFrozenGenerator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthFrozenGenerator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowFrozenGenerator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupFrozenGenerator = fmt.Errorf("proto: unexpected end of group")
)
