// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/tool/switch_container.proto

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

type SwitchContainerOptions struct {
	ContainedNode   []*framework.CalculatorGraphConfig_Node `protobuf:"bytes,2,rep,name=contained_node,json=containedNode" json:"contained_node,omitempty"`
	Select          int32                                   `protobuf:"varint,3,opt,name=select" json:"select"`
	Enable          bool                                    `protobuf:"varint,4,opt,name=enable" json:"enable"`
	SynchronizeIo   bool                                    `protobuf:"varint,5,opt,name=synchronize_io,json=synchronizeIo" json:"synchronize_io"`
	AsyncSelection  bool                                    `protobuf:"varint,6,opt,name=async_selection,json=asyncSelection" json:"async_selection"`
	TickInputStream []string                                `protobuf:"bytes,7,rep,name=tick_input_stream,json=tickInputStream" json:"tick_input_stream,omitempty"`
}

func (m *SwitchContainerOptions) Reset()      { *m = SwitchContainerOptions{} }
func (*SwitchContainerOptions) ProtoMessage() {}
func (*SwitchContainerOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_7a95e01f1905f4c7, []int{0}
}
func (m *SwitchContainerOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *SwitchContainerOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_SwitchContainerOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *SwitchContainerOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_SwitchContainerOptions.Merge(m, src)
}
func (m *SwitchContainerOptions) XXX_Size() int {
	return m.Size()
}
func (m *SwitchContainerOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_SwitchContainerOptions.DiscardUnknown(m)
}

var xxx_messageInfo_SwitchContainerOptions proto.InternalMessageInfo

func (m *SwitchContainerOptions) GetContainedNode() []*framework.CalculatorGraphConfig_Node {
	if m != nil {
		return m.ContainedNode
	}
	return nil
}

func (m *SwitchContainerOptions) GetSelect() int32 {
	if m != nil {
		return m.Select
	}
	return 0
}

func (m *SwitchContainerOptions) GetEnable() bool {
	if m != nil {
		return m.Enable
	}
	return false
}

func (m *SwitchContainerOptions) GetSynchronizeIo() bool {
	if m != nil {
		return m.SynchronizeIo
	}
	return false
}

func (m *SwitchContainerOptions) GetAsyncSelection() bool {
	if m != nil {
		return m.AsyncSelection
	}
	return false
}

func (m *SwitchContainerOptions) GetTickInputStream() []string {
	if m != nil {
		return m.TickInputStream
	}
	return nil
}

var E_SwitchContainerOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*SwitchContainerOptions)(nil),
	Field:         345967970,
	Name:          "mediapipe.SwitchContainerOptions.ext",
	Tag:           "bytes,345967970,opt,name=ext",
	Filename:      "mediapipe/framework/tool/switch_container.proto",
}

func init() {
	proto.RegisterExtension(E_SwitchContainerOptions_Ext)
	proto.RegisterType((*SwitchContainerOptions)(nil), "mediapipe.SwitchContainerOptions")
}

func init() {
	proto.RegisterFile("mediapipe/framework/tool/switch_container.proto", fileDescriptor_7a95e01f1905f4c7)
}

var fileDescriptor_7a95e01f1905f4c7 = []byte{
	// 420 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x92, 0xc1, 0x6b, 0xd4, 0x40,
	0x14, 0xc6, 0x33, 0xcd, 0xb6, 0xb6, 0x53, 0xba, 0xd5, 0x20, 0x12, 0x4a, 0x19, 0xa3, 0x28, 0x04,
	0xc5, 0x04, 0x16, 0xff, 0x82, 0xed, 0x41, 0x2a, 0xa2, 0x92, 0xc5, 0x8b, 0x97, 0x30, 0x9d, 0x4c,
	0xb3, 0x43, 0x93, 0x79, 0x21, 0x99, 0xa5, 0xea, 0xc9, 0x9b, 0x57, 0xcf, 0xe2, 0xd1, 0x83, 0x7f,
	0xca, 0x1e, 0xf7, 0xb8, 0x27, 0x71, 0x67, 0x2f, 0x1e, 0xf7, 0x0f, 0xf0, 0x20, 0x93, 0xdd, 0x84,
	0x20, 0xdb, 0xe3, 0x7c, 0xdf, 0xfb, 0x7e, 0x0f, 0xbe, 0x79, 0x38, 0xcc, 0x79, 0x22, 0x68, 0x21,
	0x0a, 0x1e, 0x5e, 0x96, 0x34, 0xe7, 0xd7, 0x50, 0x5e, 0x85, 0x0a, 0x20, 0x0b, 0xab, 0x6b, 0xa1,
	0xd8, 0x38, 0x66, 0x20, 0x15, 0x15, 0x92, 0x97, 0x41, 0x51, 0x82, 0x02, 0xe7, 0xa0, 0x0d, 0x9c,
	0x3c, 0xda, 0x96, 0x65, 0x34, 0x63, 0x93, 0x8c, 0x2a, 0xd8, 0x04, 0x1e, 0x7e, 0xb1, 0xf1, 0xbd,
	0x51, 0xcd, 0x3a, 0x6b, 0x50, 0x6f, 0x0a, 0x25, 0x40, 0x56, 0xce, 0x2b, 0xdc, 0x6f, 0xf0, 0x49,
	0x2c, 0x21, 0xe1, 0xee, 0x8e, 0x67, 0xfb, 0x87, 0x83, 0xc7, 0x41, 0x4b, 0x0e, 0xce, 0x5a, 0xde,
	0x8b, 0x92, 0x16, 0x86, 0x71, 0x29, 0xd2, 0xe0, 0x35, 0x24, 0x3c, 0x3a, 0x6a, 0xc3, 0xe6, 0xe9,
	0x9c, 0xe2, 0xbd, 0x8a, 0x67, 0x9c, 0x29, 0xd7, 0xf6, 0x90, 0xbf, 0x3b, 0xec, 0x4d, 0x7f, 0xdd,
	0xb7, 0xa2, 0x8d, 0x66, 0x5c, 0x2e, 0xe9, 0x45, 0xc6, 0xdd, 0x9e, 0x87, 0xfc, 0xfd, 0xc6, 0x5d,
	0x6b, 0xce, 0x53, 0xdc, 0xaf, 0x3e, 0x4a, 0x36, 0x2e, 0x41, 0x8a, 0x4f, 0x3c, 0x16, 0xe0, 0xee,
	0x76, 0xa6, 0x8e, 0x3a, 0xde, 0x39, 0x38, 0xcf, 0xf0, 0x31, 0x35, 0x4a, 0xbc, 0x46, 0x0b, 0x90,
	0xee, 0x5e, 0x67, 0xba, 0x5f, 0x9b, 0xa3, 0xc6, 0x73, 0x9e, 0xe0, 0x3b, 0x4a, 0xb0, 0xab, 0x58,
	0xc8, 0x62, 0xa2, 0xe2, 0x4a, 0x95, 0x9c, 0xe6, 0xee, 0x2d, 0xcf, 0xf6, 0x0f, 0xa2, 0x63, 0x63,
	0x9c, 0x1b, 0x7d, 0x54, 0xcb, 0x83, 0x77, 0xd8, 0xe6, 0x1f, 0x94, 0x73, 0xba, 0xb5, 0x80, 0x4d,
	0x6d, 0xae, 0xfe, 0xf6, 0xf7, 0x07, 0xf2, 0x90, 0x7f, 0x38, 0x78, 0xd0, 0x99, 0xdb, 0xde, 0x71,
	0x64, 0x78, 0x2f, 0x7b, 0xfb, 0xe8, 0xf6, 0xce, 0xf0, 0x3b, 0x9a, 0x2d, 0x88, 0x35, 0x5f, 0x10,
	0x6b, 0xb5, 0x20, 0xe8, 0xb3, 0x26, 0xe8, 0xa7, 0x26, 0x68, 0xaa, 0x09, 0x9a, 0x69, 0x82, 0x7e,
	0x6b, 0x82, 0xfe, 0x68, 0x62, 0xad, 0x34, 0x41, 0x5f, 0x97, 0xc4, 0x9a, 0x2d, 0x89, 0x35, 0x5f,
	0x12, 0x0b, 0x9f, 0x30, 0xc8, 0x83, 0x14, 0x20, 0xcd, 0x78, 0x67, 0x5d, 0xfd, 0xb9, 0xc3, 0xbb,
	0xff, 0x6d, 0x7d, 0x6b, 0xd4, 0xf7, 0xcf, 0x53, 0xa1, 0xc6, 0x93, 0x8b, 0x80, 0x41, 0x1e, 0xae,
	0x83, 0x9d, 0x33, 0xbb, 0xe9, 0xe0, 0xfe, 0x05, 0x00, 0x00, 0xff, 0xff, 0x35, 0xa9, 0xb6, 0xb4,
	0x8b, 0x02, 0x00, 0x00,
}

func (this *SwitchContainerOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*SwitchContainerOptions)
	if !ok {
		that2, ok := that.(SwitchContainerOptions)
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
	if len(this.ContainedNode) != len(that1.ContainedNode) {
		return false
	}
	for i := range this.ContainedNode {
		if !this.ContainedNode[i].Equal(that1.ContainedNode[i]) {
			return false
		}
	}
	if this.Select != that1.Select {
		return false
	}
	if this.Enable != that1.Enable {
		return false
	}
	if this.SynchronizeIo != that1.SynchronizeIo {
		return false
	}
	if this.AsyncSelection != that1.AsyncSelection {
		return false
	}
	if len(this.TickInputStream) != len(that1.TickInputStream) {
		return false
	}
	for i := range this.TickInputStream {
		if this.TickInputStream[i] != that1.TickInputStream[i] {
			return false
		}
	}
	return true
}
func (this *SwitchContainerOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 10)
	s = append(s, "&tool.SwitchContainerOptions{")
	if this.ContainedNode != nil {
		s = append(s, "ContainedNode: "+fmt.Sprintf("%#v", this.ContainedNode)+",\n")
	}
	s = append(s, "Select: "+fmt.Sprintf("%#v", this.Select)+",\n")
	s = append(s, "Enable: "+fmt.Sprintf("%#v", this.Enable)+",\n")
	s = append(s, "SynchronizeIo: "+fmt.Sprintf("%#v", this.SynchronizeIo)+",\n")
	s = append(s, "AsyncSelection: "+fmt.Sprintf("%#v", this.AsyncSelection)+",\n")
	if this.TickInputStream != nil {
		s = append(s, "TickInputStream: "+fmt.Sprintf("%#v", this.TickInputStream)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringSwitchContainer(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *SwitchContainerOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *SwitchContainerOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *SwitchContainerOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.TickInputStream) > 0 {
		for iNdEx := len(m.TickInputStream) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.TickInputStream[iNdEx])
			copy(dAtA[i:], m.TickInputStream[iNdEx])
			i = encodeVarintSwitchContainer(dAtA, i, uint64(len(m.TickInputStream[iNdEx])))
			i--
			dAtA[i] = 0x3a
		}
	}
	i--
	if m.AsyncSelection {
		dAtA[i] = 1
	} else {
		dAtA[i] = 0
	}
	i--
	dAtA[i] = 0x30
	i--
	if m.SynchronizeIo {
		dAtA[i] = 1
	} else {
		dAtA[i] = 0
	}
	i--
	dAtA[i] = 0x28
	i--
	if m.Enable {
		dAtA[i] = 1
	} else {
		dAtA[i] = 0
	}
	i--
	dAtA[i] = 0x20
	i = encodeVarintSwitchContainer(dAtA, i, uint64(m.Select))
	i--
	dAtA[i] = 0x18
	if len(m.ContainedNode) > 0 {
		for iNdEx := len(m.ContainedNode) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.ContainedNode[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintSwitchContainer(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0x12
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintSwitchContainer(dAtA []byte, offset int, v uint64) int {
	offset -= sovSwitchContainer(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *SwitchContainerOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.ContainedNode) > 0 {
		for _, e := range m.ContainedNode {
			l = e.Size()
			n += 1 + l + sovSwitchContainer(uint64(l))
		}
	}
	n += 1 + sovSwitchContainer(uint64(m.Select))
	n += 2
	n += 2
	n += 2
	if len(m.TickInputStream) > 0 {
		for _, s := range m.TickInputStream {
			l = len(s)
			n += 1 + l + sovSwitchContainer(uint64(l))
		}
	}
	return n
}

func sovSwitchContainer(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozSwitchContainer(x uint64) (n int) {
	return sovSwitchContainer(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *SwitchContainerOptions) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForContainedNode := "[]*CalculatorGraphConfig_Node{"
	for _, f := range this.ContainedNode {
		repeatedStringForContainedNode += strings.Replace(fmt.Sprintf("%v", f), "CalculatorGraphConfig_Node", "framework.CalculatorGraphConfig_Node", 1) + ","
	}
	repeatedStringForContainedNode += "}"
	s := strings.Join([]string{`&SwitchContainerOptions{`,
		`ContainedNode:` + repeatedStringForContainedNode + `,`,
		`Select:` + fmt.Sprintf("%v", this.Select) + `,`,
		`Enable:` + fmt.Sprintf("%v", this.Enable) + `,`,
		`SynchronizeIo:` + fmt.Sprintf("%v", this.SynchronizeIo) + `,`,
		`AsyncSelection:` + fmt.Sprintf("%v", this.AsyncSelection) + `,`,
		`TickInputStream:` + fmt.Sprintf("%v", this.TickInputStream) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringSwitchContainer(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *SwitchContainerOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowSwitchContainer
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
			return fmt.Errorf("proto: SwitchContainerOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: SwitchContainerOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ContainedNode", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSwitchContainer
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
				return ErrInvalidLengthSwitchContainer
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthSwitchContainer
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ContainedNode = append(m.ContainedNode, &framework.CalculatorGraphConfig_Node{})
			if err := m.ContainedNode[len(m.ContainedNode)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Select", wireType)
			}
			m.Select = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSwitchContainer
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Select |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Enable", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSwitchContainer
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
			m.Enable = bool(v != 0)
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field SynchronizeIo", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSwitchContainer
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
			m.SynchronizeIo = bool(v != 0)
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field AsyncSelection", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSwitchContainer
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
			m.AsyncSelection = bool(v != 0)
		case 7:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field TickInputStream", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowSwitchContainer
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
				return ErrInvalidLengthSwitchContainer
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthSwitchContainer
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.TickInputStream = append(m.TickInputStream, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipSwitchContainer(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthSwitchContainer
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
func skipSwitchContainer(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowSwitchContainer
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
					return 0, ErrIntOverflowSwitchContainer
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
					return 0, ErrIntOverflowSwitchContainer
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
				return 0, ErrInvalidLengthSwitchContainer
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupSwitchContainer
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthSwitchContainer
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthSwitchContainer        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowSwitchContainer          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupSwitchContainer = fmt.Errorf("proto: unexpected end of group")
)
