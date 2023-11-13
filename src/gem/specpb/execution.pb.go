// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/specpb/execution.proto

package specpb

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

type ExecutionSpec struct {
	Graph *framework.CalculatorGraphConfig `protobuf:"bytes,1,opt,name=graph,proto3" json:"graph,omitempty"`
}

func (m *ExecutionSpec) Reset()      { *m = ExecutionSpec{} }
func (*ExecutionSpec) ProtoMessage() {}
func (*ExecutionSpec) Descriptor() ([]byte, []int) {
	return fileDescriptor_e46fae7c9f86315d, []int{0}
}
func (m *ExecutionSpec) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ExecutionSpec) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ExecutionSpec.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ExecutionSpec) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ExecutionSpec.Merge(m, src)
}
func (m *ExecutionSpec) XXX_Size() int {
	return m.Size()
}
func (m *ExecutionSpec) XXX_DiscardUnknown() {
	xxx_messageInfo_ExecutionSpec.DiscardUnknown(m)
}

var xxx_messageInfo_ExecutionSpec proto.InternalMessageInfo

func (m *ExecutionSpec) GetGraph() *framework.CalculatorGraphConfig {
	if m != nil {
		return m.Graph
	}
	return nil
}

func init() {
	proto.RegisterType((*ExecutionSpec)(nil), "gml.gem.specpb.ExecutionSpec")
}

func init() { proto.RegisterFile("src/gem/specpb/execution.proto", fileDescriptor_e46fae7c9f86315d) }

var fileDescriptor_e46fae7c9f86315d = []byte{
	// 235 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x92, 0x2b, 0x2e, 0x4a, 0xd6,
	0x4f, 0x4f, 0xcd, 0xd5, 0x2f, 0x2e, 0x48, 0x4d, 0x2e, 0x48, 0xd2, 0x4f, 0xad, 0x48, 0x4d, 0x2e,
	0x2d, 0xc9, 0xcc, 0xcf, 0xd3, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0x4b, 0xcf, 0xcd, 0xd1,
	0x4b, 0x4f, 0xcd, 0xd5, 0x83, 0xc8, 0x4b, 0xa9, 0xe4, 0xa6, 0xa6, 0x64, 0x26, 0x16, 0x64, 0x16,
	0xa4, 0xea, 0xa7, 0x15, 0x25, 0xe6, 0xa6, 0x96, 0xe7, 0x17, 0x65, 0xeb, 0x27, 0x27, 0xe6, 0x24,
	0x97, 0xe6, 0x24, 0x96, 0xe4, 0x17, 0x41, 0x74, 0x29, 0xb9, 0x73, 0xf1, 0xba, 0xc2, 0x0c, 0x0a,
	0x2e, 0x48, 0x4d, 0x16, 0x32, 0xe3, 0x62, 0x4d, 0x2f, 0x4a, 0x2c, 0xc8, 0x90, 0x60, 0x54, 0x60,
	0xd4, 0xe0, 0x36, 0x52, 0xd0, 0x83, 0x1b, 0xa3, 0xe7, 0x0c, 0xd7, 0xec, 0x0e, 0x52, 0xe1, 0x9c,
	0x9f, 0x97, 0x96, 0x99, 0x1e, 0x04, 0x51, 0xee, 0x94, 0x70, 0xe1, 0xa1, 0x1c, 0xc3, 0x8d, 0x87,
	0x72, 0x0c, 0x1f, 0x1e, 0xca, 0x31, 0x36, 0x3c, 0x92, 0x63, 0x5c, 0xf1, 0x48, 0x8e, 0xf1, 0xc4,
	0x23, 0x39, 0xc6, 0x0b, 0x8f, 0xe4, 0x18, 0x1f, 0x3c, 0x92, 0x63, 0x7c, 0xf1, 0x48, 0x8e, 0xe1,
	0xc3, 0x23, 0x39, 0xc6, 0x09, 0x8f, 0xe5, 0x18, 0x2e, 0x3c, 0x96, 0x63, 0xb8, 0xf1, 0x58, 0x8e,
	0x21, 0x4a, 0x2b, 0x3d, 0x33, 0x37, 0x27, 0xb5, 0x24, 0x27, 0x31, 0xa9, 0x58, 0x2f, 0x31, 0x53,
	0x1f, 0xc2, 0xd3, 0x47, 0xf5, 0xa9, 0x35, 0x84, 0x4a, 0x62, 0x03, 0xbb, 0xd8, 0x18, 0x10, 0x00,
	0x00, 0xff, 0xff, 0xef, 0x00, 0x44, 0x9b, 0x09, 0x01, 0x00, 0x00,
}

func (this *ExecutionSpec) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ExecutionSpec)
	if !ok {
		that2, ok := that.(ExecutionSpec)
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
	if !this.Graph.Equal(that1.Graph) {
		return false
	}
	return true
}
func (this *ExecutionSpec) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&specpb.ExecutionSpec{")
	if this.Graph != nil {
		s = append(s, "Graph: "+fmt.Sprintf("%#v", this.Graph)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringExecution(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *ExecutionSpec) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ExecutionSpec) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ExecutionSpec) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Graph != nil {
		{
			size, err := m.Graph.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintExecution(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintExecution(dAtA []byte, offset int, v uint64) int {
	offset -= sovExecution(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ExecutionSpec) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Graph != nil {
		l = m.Graph.Size()
		n += 1 + l + sovExecution(uint64(l))
	}
	return n
}

func sovExecution(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozExecution(x uint64) (n int) {
	return sovExecution(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ExecutionSpec) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ExecutionSpec{`,
		`Graph:` + strings.Replace(fmt.Sprintf("%v", this.Graph), "CalculatorGraphConfig", "framework.CalculatorGraphConfig", 1) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringExecution(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ExecutionSpec) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowExecution
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
			return fmt.Errorf("proto: ExecutionSpec: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExecutionSpec: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Graph", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowExecution
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
				return ErrInvalidLengthExecution
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthExecution
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Graph == nil {
				m.Graph = &framework.CalculatorGraphConfig{}
			}
			if err := m.Graph.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipExecution(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthExecution
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
func skipExecution(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowExecution
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
					return 0, ErrIntOverflowExecution
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
					return 0, ErrIntOverflowExecution
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
				return 0, ErrInvalidLengthExecution
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupExecution
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthExecution
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthExecution        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowExecution          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupExecution = fmt.Errorf("proto: unexpected end of group")
)
