// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/common/typespb/dag.proto

package typespb

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

type DAG struct {
	Nodes []*DAG_DAGNode `protobuf:"bytes,1,rep,name=nodes,proto3" json:"nodes,omitempty"`
}

func (m *DAG) Reset()      { *m = DAG{} }
func (*DAG) ProtoMessage() {}
func (*DAG) Descriptor() ([]byte, []int) {
	return fileDescriptor_ee12eaf822b3c2e8, []int{0}
}
func (m *DAG) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *DAG) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_DAG.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *DAG) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DAG.Merge(m, src)
}
func (m *DAG) XXX_Size() int {
	return m.Size()
}
func (m *DAG) XXX_DiscardUnknown() {
	xxx_messageInfo_DAG.DiscardUnknown(m)
}

var xxx_messageInfo_DAG proto.InternalMessageInfo

func (m *DAG) GetNodes() []*DAG_DAGNode {
	if m != nil {
		return m.Nodes
	}
	return nil
}

type DAG_DAGNode struct {
	Id             uint64   `protobuf:"varint,1,opt,name=id,proto3" json:"id,omitempty"`
	SortedParents  []uint64 `protobuf:"varint,4,rep,packed,name=sorted_parents,json=sortedParents,proto3" json:"sorted_parents,omitempty"`
	SortedChildren []uint64 `protobuf:"varint,3,rep,packed,name=sorted_children,json=sortedChildren,proto3" json:"sorted_children,omitempty"`
}

func (m *DAG_DAGNode) Reset()      { *m = DAG_DAGNode{} }
func (*DAG_DAGNode) ProtoMessage() {}
func (*DAG_DAGNode) Descriptor() ([]byte, []int) {
	return fileDescriptor_ee12eaf822b3c2e8, []int{0, 0}
}
func (m *DAG_DAGNode) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *DAG_DAGNode) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_DAG_DAGNode.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *DAG_DAGNode) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DAG_DAGNode.Merge(m, src)
}
func (m *DAG_DAGNode) XXX_Size() int {
	return m.Size()
}
func (m *DAG_DAGNode) XXX_DiscardUnknown() {
	xxx_messageInfo_DAG_DAGNode.DiscardUnknown(m)
}

var xxx_messageInfo_DAG_DAGNode proto.InternalMessageInfo

func (m *DAG_DAGNode) GetId() uint64 {
	if m != nil {
		return m.Id
	}
	return 0
}

func (m *DAG_DAGNode) GetSortedParents() []uint64 {
	if m != nil {
		return m.SortedParents
	}
	return nil
}

func (m *DAG_DAGNode) GetSortedChildren() []uint64 {
	if m != nil {
		return m.SortedChildren
	}
	return nil
}

func init() {
	proto.RegisterType((*DAG)(nil), "gml.types.DAG")
	proto.RegisterType((*DAG_DAGNode)(nil), "gml.types.DAG.DAGNode")
}

func init() { proto.RegisterFile("src/common/typespb/dag.proto", fileDescriptor_ee12eaf822b3c2e8) }

var fileDescriptor_ee12eaf822b3c2e8 = []byte{
	// 263 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x92, 0x29, 0x2e, 0x4a, 0xd6,
	0x4f, 0xce, 0xcf, 0xcd, 0xcd, 0xcf, 0xd3, 0x2f, 0xa9, 0x2c, 0x48, 0x2d, 0x2e, 0x48, 0xd2, 0x4f,
	0x49, 0x4c, 0xd7, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0x4c, 0xcf, 0xcd, 0xd1, 0x03, 0x0b,
	0x2b, 0xcd, 0x63, 0xe4, 0x62, 0x76, 0x71, 0x74, 0x17, 0xd2, 0xe1, 0x62, 0xcd, 0xcb, 0x4f, 0x49,
	0x2d, 0x96, 0x60, 0x54, 0x60, 0xd6, 0xe0, 0x36, 0x12, 0xd3, 0x83, 0x2b, 0xd1, 0x73, 0x71, 0x74,
	0x07, 0x61, 0xbf, 0xfc, 0x94, 0xd4, 0x20, 0x88, 0x22, 0xa9, 0x4c, 0x2e, 0x76, 0xa8, 0x88, 0x10,
	0x1f, 0x17, 0x53, 0x66, 0x8a, 0x04, 0xa3, 0x02, 0xa3, 0x06, 0x4b, 0x10, 0x53, 0x66, 0x8a, 0x90,
	0x2a, 0x17, 0x5f, 0x71, 0x7e, 0x51, 0x49, 0x6a, 0x4a, 0x7c, 0x41, 0x62, 0x51, 0x6a, 0x5e, 0x49,
	0xb1, 0x04, 0x8b, 0x02, 0xb3, 0x06, 0x4b, 0x10, 0x2f, 0x44, 0x34, 0x00, 0x22, 0x28, 0xa4, 0xce,
	0xc5, 0x0f, 0x55, 0x96, 0x9c, 0x91, 0x99, 0x93, 0x52, 0x94, 0x9a, 0x27, 0xc1, 0x0c, 0x56, 0x07,
	0xd5, 0xed, 0x0c, 0x15, 0x75, 0x4a, 0xbd, 0xf0, 0x50, 0x8e, 0xe1, 0xc6, 0x43, 0x39, 0x86, 0x0f,
	0x0f, 0xe5, 0x18, 0x1b, 0x1e, 0xc9, 0x31, 0xae, 0x78, 0x24, 0xc7, 0x78, 0xe2, 0x91, 0x1c, 0xe3,
	0x85, 0x47, 0x72, 0x8c, 0x0f, 0x1e, 0xc9, 0x31, 0xbe, 0x78, 0x24, 0xc7, 0xf0, 0xe1, 0x91, 0x1c,
	0xe3, 0x84, 0xc7, 0x72, 0x0c, 0x17, 0x1e, 0xcb, 0x31, 0xdc, 0x78, 0x2c, 0xc7, 0x10, 0xa5, 0x9f,
	0x9e, 0x99, 0x9b, 0x93, 0x5a, 0x92, 0x93, 0x98, 0x54, 0xac, 0x97, 0x98, 0x09, 0xe5, 0xe9, 0x63,
	0x06, 0x8a, 0x35, 0x94, 0x4e, 0x62, 0x03, 0x87, 0x8c, 0x31, 0x20, 0x00, 0x00, 0xff, 0xff, 0xd7,
	0xfa, 0x0f, 0x00, 0x39, 0x01, 0x00, 0x00,
}

func (this *DAG) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*DAG)
	if !ok {
		that2, ok := that.(DAG)
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
	if len(this.Nodes) != len(that1.Nodes) {
		return false
	}
	for i := range this.Nodes {
		if !this.Nodes[i].Equal(that1.Nodes[i]) {
			return false
		}
	}
	return true
}
func (this *DAG_DAGNode) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*DAG_DAGNode)
	if !ok {
		that2, ok := that.(DAG_DAGNode)
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
	if this.Id != that1.Id {
		return false
	}
	if len(this.SortedParents) != len(that1.SortedParents) {
		return false
	}
	for i := range this.SortedParents {
		if this.SortedParents[i] != that1.SortedParents[i] {
			return false
		}
	}
	if len(this.SortedChildren) != len(that1.SortedChildren) {
		return false
	}
	for i := range this.SortedChildren {
		if this.SortedChildren[i] != that1.SortedChildren[i] {
			return false
		}
	}
	return true
}
func (this *DAG) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&typespb.DAG{")
	if this.Nodes != nil {
		s = append(s, "Nodes: "+fmt.Sprintf("%#v", this.Nodes)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *DAG_DAGNode) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&typespb.DAG_DAGNode{")
	s = append(s, "Id: "+fmt.Sprintf("%#v", this.Id)+",\n")
	s = append(s, "SortedParents: "+fmt.Sprintf("%#v", this.SortedParents)+",\n")
	s = append(s, "SortedChildren: "+fmt.Sprintf("%#v", this.SortedChildren)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringDag(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *DAG) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *DAG) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *DAG) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Nodes) > 0 {
		for iNdEx := len(m.Nodes) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.Nodes[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintDag(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func (m *DAG_DAGNode) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *DAG_DAGNode) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *DAG_DAGNode) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.SortedParents) > 0 {
		dAtA2 := make([]byte, len(m.SortedParents)*10)
		var j1 int
		for _, num := range m.SortedParents {
			for num >= 1<<7 {
				dAtA2[j1] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j1++
			}
			dAtA2[j1] = uint8(num)
			j1++
		}
		i -= j1
		copy(dAtA[i:], dAtA2[:j1])
		i = encodeVarintDag(dAtA, i, uint64(j1))
		i--
		dAtA[i] = 0x22
	}
	if len(m.SortedChildren) > 0 {
		dAtA4 := make([]byte, len(m.SortedChildren)*10)
		var j3 int
		for _, num := range m.SortedChildren {
			for num >= 1<<7 {
				dAtA4[j3] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j3++
			}
			dAtA4[j3] = uint8(num)
			j3++
		}
		i -= j3
		copy(dAtA[i:], dAtA4[:j3])
		i = encodeVarintDag(dAtA, i, uint64(j3))
		i--
		dAtA[i] = 0x1a
	}
	if m.Id != 0 {
		i = encodeVarintDag(dAtA, i, uint64(m.Id))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintDag(dAtA []byte, offset int, v uint64) int {
	offset -= sovDag(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *DAG) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Nodes) > 0 {
		for _, e := range m.Nodes {
			l = e.Size()
			n += 1 + l + sovDag(uint64(l))
		}
	}
	return n
}

func (m *DAG_DAGNode) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Id != 0 {
		n += 1 + sovDag(uint64(m.Id))
	}
	if len(m.SortedChildren) > 0 {
		l = 0
		for _, e := range m.SortedChildren {
			l += sovDag(uint64(e))
		}
		n += 1 + sovDag(uint64(l)) + l
	}
	if len(m.SortedParents) > 0 {
		l = 0
		for _, e := range m.SortedParents {
			l += sovDag(uint64(e))
		}
		n += 1 + sovDag(uint64(l)) + l
	}
	return n
}

func sovDag(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozDag(x uint64) (n int) {
	return sovDag(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *DAG) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForNodes := "[]*DAG_DAGNode{"
	for _, f := range this.Nodes {
		repeatedStringForNodes += strings.Replace(fmt.Sprintf("%v", f), "DAG_DAGNode", "DAG_DAGNode", 1) + ","
	}
	repeatedStringForNodes += "}"
	s := strings.Join([]string{`&DAG{`,
		`Nodes:` + repeatedStringForNodes + `,`,
		`}`,
	}, "")
	return s
}
func (this *DAG_DAGNode) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&DAG_DAGNode{`,
		`Id:` + fmt.Sprintf("%v", this.Id) + `,`,
		`SortedChildren:` + fmt.Sprintf("%v", this.SortedChildren) + `,`,
		`SortedParents:` + fmt.Sprintf("%v", this.SortedParents) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringDag(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *DAG) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowDag
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
			return fmt.Errorf("proto: DAG: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: DAG: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Nodes", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDag
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
				return ErrInvalidLengthDag
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthDag
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Nodes = append(m.Nodes, &DAG_DAGNode{})
			if err := m.Nodes[len(m.Nodes)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipDag(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthDag
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
func (m *DAG_DAGNode) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowDag
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
			return fmt.Errorf("proto: DAGNode: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: DAGNode: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Id", wireType)
			}
			m.Id = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDag
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Id |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType == 0 {
				var v uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDag
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					v |= uint64(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.SortedChildren = append(m.SortedChildren, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDag
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthDag
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthDag
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range dAtA[iNdEx:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.SortedChildren) == 0 {
					m.SortedChildren = make([]uint64, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowDag
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						v |= uint64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.SortedChildren = append(m.SortedChildren, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field SortedChildren", wireType)
			}
		case 4:
			if wireType == 0 {
				var v uint64
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDag
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					v |= uint64(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				m.SortedParents = append(m.SortedParents, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDag
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthDag
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthDag
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range dAtA[iNdEx:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.SortedParents) == 0 {
					m.SortedParents = make([]uint64, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v uint64
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowDag
						}
						if iNdEx >= l {
							return io.ErrUnexpectedEOF
						}
						b := dAtA[iNdEx]
						iNdEx++
						v |= uint64(b&0x7F) << shift
						if b < 0x80 {
							break
						}
					}
					m.SortedParents = append(m.SortedParents, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field SortedParents", wireType)
			}
		default:
			iNdEx = preIndex
			skippy, err := skipDag(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthDag
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
func skipDag(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowDag
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
					return 0, ErrIntOverflowDag
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
					return 0, ErrIntOverflowDag
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
				return 0, ErrInvalidLengthDag
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupDag
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthDag
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthDag        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowDag          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupDag = fmt.Errorf("proto: unexpected end of group")
)
