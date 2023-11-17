// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/util/label_map.proto

package util

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

type LabelMapItem struct {
	Name        string   `protobuf:"bytes,1,opt,name=name" json:"name"`
	DisplayName string   `protobuf:"bytes,2,opt,name=display_name,json=displayName" json:"display_name"`
	ChildName   []string `protobuf:"bytes,3,rep,name=child_name,json=childName" json:"child_name,omitempty"`
}

func (m *LabelMapItem) Reset()      { *m = LabelMapItem{} }
func (*LabelMapItem) ProtoMessage() {}
func (*LabelMapItem) Descriptor() ([]byte, []int) {
	return fileDescriptor_30314dcba0d58f0f, []int{0}
}
func (m *LabelMapItem) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *LabelMapItem) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_LabelMapItem.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *LabelMapItem) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LabelMapItem.Merge(m, src)
}
func (m *LabelMapItem) XXX_Size() int {
	return m.Size()
}
func (m *LabelMapItem) XXX_DiscardUnknown() {
	xxx_messageInfo_LabelMapItem.DiscardUnknown(m)
}

var xxx_messageInfo_LabelMapItem proto.InternalMessageInfo

func (m *LabelMapItem) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *LabelMapItem) GetDisplayName() string {
	if m != nil {
		return m.DisplayName
	}
	return ""
}

func (m *LabelMapItem) GetChildName() []string {
	if m != nil {
		return m.ChildName
	}
	return nil
}

func init() {
	proto.RegisterType((*LabelMapItem)(nil), "mediapipe.LabelMapItem")
}

func init() { proto.RegisterFile("mediapipe/util/label_map.proto", fileDescriptor_30314dcba0d58f0f) }

var fileDescriptor_30314dcba0d58f0f = []byte{
	// 247 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x54, 0x90, 0x31, 0x4e, 0xc3, 0x30,
	0x14, 0x86, 0xfd, 0x68, 0x17, 0x9b, 0xb2, 0x64, 0xea, 0xc2, 0x6b, 0xc5, 0x42, 0xc5, 0x60, 0xdf,
	0x21, 0x1b, 0x12, 0x20, 0xd4, 0x91, 0xa5, 0x72, 0x13, 0x2b, 0xb5, 0x64, 0x63, 0x0b, 0xdc, 0x81,
	0x8d, 0x23, 0xf4, 0x18, 0x1c, 0xa5, 0x63, 0xc6, 0x4e, 0x88, 0x38, 0x0b, 0x63, 0x8f, 0x80, 0x9c,
	0x40, 0x80, 0xd1, 0xff, 0xf7, 0x59, 0x7a, 0xff, 0xcf, 0xd0, 0xaa, 0x52, 0x4b, 0xaf, 0xbd, 0x12,
	0xdb, 0xa0, 0x8d, 0x30, 0x72, 0xad, 0xcc, 0xca, 0x4a, 0xcf, 0xfd, 0x93, 0x0b, 0x2e, 0xa3, 0x03,
	0xbf, 0xf0, 0x6c, 0x72, 0x93, 0xe8, 0xad, 0xf4, 0xd7, 0x41, 0xd9, 0x6c, 0xca, 0xc6, 0x8f, 0xd2,
	0xaa, 0x29, 0xcc, 0x61, 0x41, 0xf3, 0xf1, 0xfe, 0x7d, 0x46, 0x96, 0x5d, 0x92, 0x5d, 0xb2, 0x49,
	0xa9, 0x9f, 0xbd, 0x91, 0x2f, 0xab, 0xce, 0x38, 0xf9, 0x63, 0x9c, 0x7e, 0x93, 0xbb, 0x24, 0x9e,
	0x33, 0x56, 0x6c, 0xb4, 0x29, 0x7b, 0x6d, 0x34, 0x1f, 0x2d, 0xe8, 0x92, 0x76, 0x49, 0xc2, 0xf9,
	0x0e, 0xea, 0x06, 0xc9, 0xa1, 0x41, 0x72, 0x6c, 0x10, 0x5e, 0x23, 0xc2, 0x5b, 0x44, 0xd8, 0x47,
	0x84, 0x3a, 0x22, 0x7c, 0x44, 0x84, 0xcf, 0x88, 0xe4, 0x18, 0x11, 0x76, 0x2d, 0x92, 0xba, 0x45,
	0x72, 0x68, 0x91, 0xb0, 0x59, 0xe1, 0x2c, 0xaf, 0x9c, 0xab, 0x8c, 0xe2, 0xc3, 0xf9, 0x3c, 0xd5,
	0xeb, 0x4b, 0xe5, 0x67, 0x3f, 0x3d, 0xee, 0xd3, 0xf3, 0xe1, 0xaa, 0xd2, 0x61, 0xb3, 0x5d, 0xf3,
	0xc2, 0x59, 0xd1, 0x7f, 0x13, 0xbf, 0xab, 0xfc, 0xdf, 0xe7, 0x2b, 0x00, 0x00, 0xff, 0xff, 0xf7,
	0x3d, 0xb5, 0x86, 0x30, 0x01, 0x00, 0x00,
}

func (this *LabelMapItem) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*LabelMapItem)
	if !ok {
		that2, ok := that.(LabelMapItem)
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
	if this.Name != that1.Name {
		return false
	}
	if this.DisplayName != that1.DisplayName {
		return false
	}
	if len(this.ChildName) != len(that1.ChildName) {
		return false
	}
	for i := range this.ChildName {
		if this.ChildName[i] != that1.ChildName[i] {
			return false
		}
	}
	return true
}
func (this *LabelMapItem) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&util.LabelMapItem{")
	s = append(s, "Name: "+fmt.Sprintf("%#v", this.Name)+",\n")
	s = append(s, "DisplayName: "+fmt.Sprintf("%#v", this.DisplayName)+",\n")
	if this.ChildName != nil {
		s = append(s, "ChildName: "+fmt.Sprintf("%#v", this.ChildName)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringLabelMap(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *LabelMapItem) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *LabelMapItem) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *LabelMapItem) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.ChildName) > 0 {
		for iNdEx := len(m.ChildName) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.ChildName[iNdEx])
			copy(dAtA[i:], m.ChildName[iNdEx])
			i = encodeVarintLabelMap(dAtA, i, uint64(len(m.ChildName[iNdEx])))
			i--
			dAtA[i] = 0x1a
		}
	}
	i -= len(m.DisplayName)
	copy(dAtA[i:], m.DisplayName)
	i = encodeVarintLabelMap(dAtA, i, uint64(len(m.DisplayName)))
	i--
	dAtA[i] = 0x12
	i -= len(m.Name)
	copy(dAtA[i:], m.Name)
	i = encodeVarintLabelMap(dAtA, i, uint64(len(m.Name)))
	i--
	dAtA[i] = 0xa
	return len(dAtA) - i, nil
}

func encodeVarintLabelMap(dAtA []byte, offset int, v uint64) int {
	offset -= sovLabelMap(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *LabelMapItem) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Name)
	n += 1 + l + sovLabelMap(uint64(l))
	l = len(m.DisplayName)
	n += 1 + l + sovLabelMap(uint64(l))
	if len(m.ChildName) > 0 {
		for _, s := range m.ChildName {
			l = len(s)
			n += 1 + l + sovLabelMap(uint64(l))
		}
	}
	return n
}

func sovLabelMap(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozLabelMap(x uint64) (n int) {
	return sovLabelMap(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *LabelMapItem) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&LabelMapItem{`,
		`Name:` + fmt.Sprintf("%v", this.Name) + `,`,
		`DisplayName:` + fmt.Sprintf("%v", this.DisplayName) + `,`,
		`ChildName:` + fmt.Sprintf("%v", this.ChildName) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringLabelMap(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *LabelMapItem) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLabelMap
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
			return fmt.Errorf("proto: LabelMapItem: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: LabelMapItem: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelMap
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
				return ErrInvalidLengthLabelMap
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthLabelMap
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field DisplayName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelMap
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
				return ErrInvalidLengthLabelMap
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthLabelMap
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.DisplayName = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ChildName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelMap
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
				return ErrInvalidLengthLabelMap
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthLabelMap
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ChildName = append(m.ChildName, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipLabelMap(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthLabelMap
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
func skipLabelMap(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowLabelMap
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
					return 0, ErrIntOverflowLabelMap
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
					return 0, ErrIntOverflowLabelMap
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
				return 0, ErrInvalidLengthLabelMap
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupLabelMap
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthLabelMap
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthLabelMap        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowLabelMap          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupLabelMap = fmt.Errorf("proto: unexpected end of group")
)