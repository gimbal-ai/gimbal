// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/util/detection_label_id_to_text_calculator.proto

package util

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	github_com_gogo_protobuf_sortkeys "github.com/gogo/protobuf/sortkeys"
	framework "github.com/google/mediapipe/mediapipe/framework"
	util "github.com/google/mediapipe/mediapipe/util"
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

type DetectionLabelIdToTextCalculatorOptions struct {
	LabelMapPath string                       `protobuf:"bytes,1,opt,name=label_map_path,json=labelMapPath" json:"label_map_path"`
	Label        []string                     `protobuf:"bytes,2,rep,name=label" json:"label,omitempty"`
	KeepLabelId  bool                         `protobuf:"varint,3,opt,name=keep_label_id,json=keepLabelId" json:"keep_label_id"`
	LabelItems   map[int64]*util.LabelMapItem `protobuf:"bytes,4,rep,name=label_items,json=labelItems" json:"label_items,omitempty" protobuf_key:"varint,1,opt,name=key" protobuf_val:"bytes,2,opt,name=value"`
}

func (m *DetectionLabelIdToTextCalculatorOptions) Reset() {
	*m = DetectionLabelIdToTextCalculatorOptions{}
}
func (*DetectionLabelIdToTextCalculatorOptions) ProtoMessage() {}
func (*DetectionLabelIdToTextCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_0edc55abfb21cf50, []int{0}
}
func (m *DetectionLabelIdToTextCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *DetectionLabelIdToTextCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_DetectionLabelIdToTextCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *DetectionLabelIdToTextCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DetectionLabelIdToTextCalculatorOptions.Merge(m, src)
}
func (m *DetectionLabelIdToTextCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *DetectionLabelIdToTextCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_DetectionLabelIdToTextCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_DetectionLabelIdToTextCalculatorOptions proto.InternalMessageInfo

func (m *DetectionLabelIdToTextCalculatorOptions) GetLabelMapPath() string {
	if m != nil {
		return m.LabelMapPath
	}
	return ""
}

func (m *DetectionLabelIdToTextCalculatorOptions) GetLabel() []string {
	if m != nil {
		return m.Label
	}
	return nil
}

func (m *DetectionLabelIdToTextCalculatorOptions) GetKeepLabelId() bool {
	if m != nil {
		return m.KeepLabelId
	}
	return false
}

func (m *DetectionLabelIdToTextCalculatorOptions) GetLabelItems() map[int64]*util.LabelMapItem {
	if m != nil {
		return m.LabelItems
	}
	return nil
}

var E_DetectionLabelIdToTextCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*DetectionLabelIdToTextCalculatorOptions)(nil),
	Field:         251889072,
	Name:          "mediapipe.DetectionLabelIdToTextCalculatorOptions.ext",
	Tag:           "bytes,251889072,opt,name=ext",
	Filename:      "mediapipe/calculators/util/detection_label_id_to_text_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_DetectionLabelIdToTextCalculatorOptions_Ext)
	proto.RegisterType((*DetectionLabelIdToTextCalculatorOptions)(nil), "mediapipe.DetectionLabelIdToTextCalculatorOptions")
	proto.RegisterMapType((map[int64]*util.LabelMapItem)(nil), "mediapipe.DetectionLabelIdToTextCalculatorOptions.LabelItemsEntry")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/util/detection_label_id_to_text_calculator.proto", fileDescriptor_0edc55abfb21cf50)
}

var fileDescriptor_0edc55abfb21cf50 = []byte{
	// 414 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x92, 0x4d, 0xcb, 0xd3, 0x40,
	0x14, 0x85, 0x33, 0xcd, 0xfb, 0x82, 0x9d, 0xf8, 0x01, 0x83, 0x68, 0x28, 0x32, 0x06, 0x11, 0x0c,
	0x82, 0x09, 0x64, 0x21, 0xe2, 0xb2, 0x7e, 0x80, 0x50, 0x51, 0x42, 0x57, 0x22, 0x84, 0x69, 0x72,
	0x6d, 0x43, 0x27, 0x9d, 0x21, 0x9d, 0x68, 0xba, 0x73, 0x2f, 0x88, 0x3f, 0xc3, 0xa5, 0xff, 0xc0,
	0x6d, 0x97, 0x5d, 0x76, 0x25, 0x36, 0xdd, 0xb8, 0xec, 0x4f, 0x90, 0x7c, 0x98, 0x94, 0x8a, 0xe0,
	0xbb, 0x9c, 0x39, 0xe7, 0x3e, 0xf7, 0x1c, 0xb8, 0xf8, 0x79, 0x02, 0x51, 0xcc, 0x64, 0x2c, 0xc1,
	0x0d, 0x19, 0x0f, 0x33, 0xce, 0x94, 0x48, 0x97, 0x6e, 0xa6, 0x62, 0xee, 0x46, 0xa0, 0x20, 0x54,
	0xb1, 0x58, 0x04, 0x9c, 0x4d, 0x80, 0x07, 0x71, 0x14, 0x28, 0x11, 0x28, 0xc8, 0x55, 0xd0, 0x79,
	0x1d, 0x99, 0x0a, 0x25, 0x48, 0xbf, 0xe5, 0x0c, 0xee, 0x76, 0xc8, 0x77, 0x29, 0x4b, 0xe0, 0x83,
	0x48, 0xe7, 0xee, 0xe9, 0xc0, 0x80, 0x76, 0xae, 0x6a, 0x59, 0xbd, 0x22, 0x61, 0xb2, 0xd6, 0xef,
	0x7c, 0xd7, 0xf1, 0xbd, 0xa7, 0x7f, 0x02, 0x8c, 0x4a, 0xf1, 0x45, 0x34, 0x16, 0x63, 0xc8, 0xd5,
	0x93, 0x16, 0xf5, 0x4a, 0x96, 0xe2, 0x92, 0xdc, 0xc7, 0x57, 0xdb, 0xf1, 0x40, 0x32, 0x35, 0x33,
	0x91, 0x85, 0xec, 0xfe, 0xf0, 0x6c, 0xfd, 0xe3, 0xb6, 0xe6, 0x5f, 0xae, 0xb4, 0x97, 0x4c, 0xbe,
	0x66, 0x6a, 0x46, 0xae, 0xe3, 0xf3, 0xea, 0x6d, 0xf6, 0x2c, 0xdd, 0xee, 0xfb, 0xf5, 0x83, 0xd8,
	0xf8, 0xca, 0x1c, 0x40, 0xb6, 0x45, 0x4d, 0xdd, 0x42, 0xf6, 0xa5, 0x06, 0x60, 0x94, 0x52, 0x13,
	0x81, 0x84, 0xd8, 0x68, 0x4c, 0x0a, 0x92, 0xa5, 0x79, 0x66, 0xe9, 0xb6, 0xe1, 0x0d, 0x9d, 0xb6,
	0x8d, 0xf3, 0x9f, 0xa1, 0x9d, 0x5a, 0x2e, 0x21, 0xcf, 0x16, 0x2a, 0x5d, 0xf9, 0x98, 0xb7, 0x1f,
	0x83, 0xb7, 0xf8, 0xda, 0x89, 0x4c, 0x6e, 0x60, 0x7d, 0x0e, 0xab, 0xaa, 0x98, 0xde, 0xe4, 0x2a,
	0x3f, 0xc8, 0x03, 0x7c, 0xfe, 0x9e, 0xf1, 0x0c, 0xcc, 0x9e, 0x85, 0x6c, 0xc3, 0xbb, 0x79, 0x94,
	0x64, 0xd4, 0xf4, 0x2e, 0x29, 0x7e, 0xed, 0x7a, 0xdc, 0x7b, 0x84, 0x3c, 0xc0, 0x3a, 0xe4, 0x8a,
	0xdc, 0x3a, 0xb2, 0xfe, 0x15, 0xcf, 0xfc, 0xf6, 0xe9, 0x73, 0x5e, 0x11, 0xbd, 0x8b, 0x77, 0xf3,
	0x4b, 0xfe, 0x90, 0x6f, 0x76, 0x54, 0xdb, 0xee, 0xa8, 0x76, 0xd8, 0x51, 0xf4, 0xb1, 0xa0, 0xe8,
	0x6b, 0x41, 0xd1, 0xba, 0xa0, 0x68, 0x53, 0x50, 0xf4, 0xb3, 0xa0, 0xe8, 0x57, 0x41, 0xb5, 0x43,
	0x41, 0xd1, 0x97, 0x3d, 0xd5, 0x36, 0x7b, 0xaa, 0x6d, 0xf7, 0x54, 0x7b, 0xf3, 0x70, 0x1a, 0xab,
	0x59, 0x36, 0x71, 0x42, 0x91, 0xb8, 0x53, 0x21, 0xa6, 0x1c, 0xdc, 0xee, 0x52, 0xfe, 0x7d, 0xac,
	0xbf, 0x03, 0x00, 0x00, 0xff, 0xff, 0x87, 0xde, 0x7d, 0x6b, 0xc9, 0x02, 0x00, 0x00,
}

func (this *DetectionLabelIdToTextCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*DetectionLabelIdToTextCalculatorOptions)
	if !ok {
		that2, ok := that.(DetectionLabelIdToTextCalculatorOptions)
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
	if this.LabelMapPath != that1.LabelMapPath {
		return false
	}
	if len(this.Label) != len(that1.Label) {
		return false
	}
	for i := range this.Label {
		if this.Label[i] != that1.Label[i] {
			return false
		}
	}
	if this.KeepLabelId != that1.KeepLabelId {
		return false
	}
	if len(this.LabelItems) != len(that1.LabelItems) {
		return false
	}
	for i := range this.LabelItems {
		if !this.LabelItems[i].Equal(that1.LabelItems[i]) {
			return false
		}
	}
	return true
}
func (this *DetectionLabelIdToTextCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 8)
	s = append(s, "&util.DetectionLabelIdToTextCalculatorOptions{")
	s = append(s, "LabelMapPath: "+fmt.Sprintf("%#v", this.LabelMapPath)+",\n")
	if this.Label != nil {
		s = append(s, "Label: "+fmt.Sprintf("%#v", this.Label)+",\n")
	}
	s = append(s, "KeepLabelId: "+fmt.Sprintf("%#v", this.KeepLabelId)+",\n")
	keysForLabelItems := make([]int64, 0, len(this.LabelItems))
	for k, _ := range this.LabelItems {
		keysForLabelItems = append(keysForLabelItems, k)
	}
	github_com_gogo_protobuf_sortkeys.Int64s(keysForLabelItems)
	mapStringForLabelItems := "map[int64]*util.LabelMapItem{"
	for _, k := range keysForLabelItems {
		mapStringForLabelItems += fmt.Sprintf("%#v: %#v,", k, this.LabelItems[k])
	}
	mapStringForLabelItems += "}"
	if this.LabelItems != nil {
		s = append(s, "LabelItems: "+mapStringForLabelItems+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringDetectionLabelIdToTextCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *DetectionLabelIdToTextCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *DetectionLabelIdToTextCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *DetectionLabelIdToTextCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.LabelItems) > 0 {
		for k := range m.LabelItems {
			v := m.LabelItems[k]
			baseI := i
			if v != nil {
				{
					size, err := v.MarshalToSizedBuffer(dAtA[:i])
					if err != nil {
						return 0, err
					}
					i -= size
					i = encodeVarintDetectionLabelIdToTextCalculator(dAtA, i, uint64(size))
				}
				i--
				dAtA[i] = 0x12
			}
			i = encodeVarintDetectionLabelIdToTextCalculator(dAtA, i, uint64(k))
			i--
			dAtA[i] = 0x8
			i = encodeVarintDetectionLabelIdToTextCalculator(dAtA, i, uint64(baseI-i))
			i--
			dAtA[i] = 0x22
		}
	}
	i--
	if m.KeepLabelId {
		dAtA[i] = 1
	} else {
		dAtA[i] = 0
	}
	i--
	dAtA[i] = 0x18
	if len(m.Label) > 0 {
		for iNdEx := len(m.Label) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.Label[iNdEx])
			copy(dAtA[i:], m.Label[iNdEx])
			i = encodeVarintDetectionLabelIdToTextCalculator(dAtA, i, uint64(len(m.Label[iNdEx])))
			i--
			dAtA[i] = 0x12
		}
	}
	i -= len(m.LabelMapPath)
	copy(dAtA[i:], m.LabelMapPath)
	i = encodeVarintDetectionLabelIdToTextCalculator(dAtA, i, uint64(len(m.LabelMapPath)))
	i--
	dAtA[i] = 0xa
	return len(dAtA) - i, nil
}

func encodeVarintDetectionLabelIdToTextCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovDetectionLabelIdToTextCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *DetectionLabelIdToTextCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.LabelMapPath)
	n += 1 + l + sovDetectionLabelIdToTextCalculator(uint64(l))
	if len(m.Label) > 0 {
		for _, s := range m.Label {
			l = len(s)
			n += 1 + l + sovDetectionLabelIdToTextCalculator(uint64(l))
		}
	}
	n += 2
	if len(m.LabelItems) > 0 {
		for k, v := range m.LabelItems {
			_ = k
			_ = v
			l = 0
			if v != nil {
				l = v.Size()
				l += 1 + sovDetectionLabelIdToTextCalculator(uint64(l))
			}
			mapEntrySize := 1 + sovDetectionLabelIdToTextCalculator(uint64(k)) + l
			n += mapEntrySize + 1 + sovDetectionLabelIdToTextCalculator(uint64(mapEntrySize))
		}
	}
	return n
}

func sovDetectionLabelIdToTextCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozDetectionLabelIdToTextCalculator(x uint64) (n int) {
	return sovDetectionLabelIdToTextCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *DetectionLabelIdToTextCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	keysForLabelItems := make([]int64, 0, len(this.LabelItems))
	for k, _ := range this.LabelItems {
		keysForLabelItems = append(keysForLabelItems, k)
	}
	github_com_gogo_protobuf_sortkeys.Int64s(keysForLabelItems)
	mapStringForLabelItems := "map[int64]*util.LabelMapItem{"
	for _, k := range keysForLabelItems {
		mapStringForLabelItems += fmt.Sprintf("%v: %v,", k, this.LabelItems[k])
	}
	mapStringForLabelItems += "}"
	s := strings.Join([]string{`&DetectionLabelIdToTextCalculatorOptions{`,
		`LabelMapPath:` + fmt.Sprintf("%v", this.LabelMapPath) + `,`,
		`Label:` + fmt.Sprintf("%v", this.Label) + `,`,
		`KeepLabelId:` + fmt.Sprintf("%v", this.KeepLabelId) + `,`,
		`LabelItems:` + mapStringForLabelItems + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringDetectionLabelIdToTextCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *DetectionLabelIdToTextCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowDetectionLabelIdToTextCalculator
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
			return fmt.Errorf("proto: DetectionLabelIdToTextCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: DetectionLabelIdToTextCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field LabelMapPath", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetectionLabelIdToTextCalculator
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
				return ErrInvalidLengthDetectionLabelIdToTextCalculator
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthDetectionLabelIdToTextCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.LabelMapPath = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Label", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetectionLabelIdToTextCalculator
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
				return ErrInvalidLengthDetectionLabelIdToTextCalculator
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthDetectionLabelIdToTextCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Label = append(m.Label, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field KeepLabelId", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetectionLabelIdToTextCalculator
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
			m.KeepLabelId = bool(v != 0)
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field LabelItems", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetectionLabelIdToTextCalculator
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
				return ErrInvalidLengthDetectionLabelIdToTextCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthDetectionLabelIdToTextCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.LabelItems == nil {
				m.LabelItems = make(map[int64]*util.LabelMapItem)
			}
			var mapkey int64
			var mapvalue *util.LabelMapItem
			for iNdEx < postIndex {
				entryPreIndex := iNdEx
				var wire uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDetectionLabelIdToTextCalculator
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
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowDetectionLabelIdToTextCalculator
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						mapkey |= int64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
				} else if fieldNum == 2 {
					var mapmsglen int
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowDetectionLabelIdToTextCalculator
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						mapmsglen |= int(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					if mapmsglen < 0 {
						return ErrInvalidLengthDetectionLabelIdToTextCalculator
					}
					postmsgIndex := iNdEx + mapmsglen
					if postmsgIndex < 0 {
						return ErrInvalidLengthDetectionLabelIdToTextCalculator
					}
					if postmsgIndex > l {
						return io.ErrUnexpectedEOF
					}
					mapvalue = &util.LabelMapItem{}
					if err := mapvalue.Unmarshal(dAtA[iNdEx:postmsgIndex]); err != nil {
						return err
					}
					iNdEx = postmsgIndex
				} else {
					iNdEx = entryPreIndex
					skippy, err := skipDetectionLabelIdToTextCalculator(dAtA[iNdEx:])
					if err != nil {
						return err
					}
					if (skippy < 0) || (iNdEx+skippy) < 0 {
						return ErrInvalidLengthDetectionLabelIdToTextCalculator
					}
					if (iNdEx + skippy) > postIndex {
						return io.ErrUnexpectedEOF
					}
					iNdEx += skippy
				}
			}
			m.LabelItems[mapkey] = mapvalue
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipDetectionLabelIdToTextCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthDetectionLabelIdToTextCalculator
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
func skipDetectionLabelIdToTextCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowDetectionLabelIdToTextCalculator
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
					return 0, ErrIntOverflowDetectionLabelIdToTextCalculator
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
					return 0, ErrIntOverflowDetectionLabelIdToTextCalculator
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
				return 0, ErrInvalidLengthDetectionLabelIdToTextCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupDetectionLabelIdToTextCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthDetectionLabelIdToTextCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthDetectionLabelIdToTextCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowDetectionLabelIdToTextCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupDetectionLabelIdToTextCalculator = fmt.Errorf("proto: unexpected end of group")
)
