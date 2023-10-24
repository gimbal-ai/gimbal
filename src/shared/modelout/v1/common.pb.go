// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/shared/modelout/v1/common.proto

package modelout

import (
	encoding_binary "encoding/binary"
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

type Label struct {
	Label string  `protobuf:"bytes,1,opt,name=label,proto3" json:"label,omitempty"`
	Score float32 `protobuf:"fixed32,2,opt,name=score,proto3" json:"score,omitempty"`
}

func (m *Label) Reset()      { *m = Label{} }
func (*Label) ProtoMessage() {}
func (*Label) Descriptor() ([]byte, []int) {
	return fileDescriptor_74a1b3e6342f71eb, []int{0}
}
func (m *Label) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Label) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Label.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Label) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Label.Merge(m, src)
}
func (m *Label) XXX_Size() int {
	return m.Size()
}
func (m *Label) XXX_DiscardUnknown() {
	xxx_messageInfo_Label.DiscardUnknown(m)
}

var xxx_messageInfo_Label proto.InternalMessageInfo

func (m *Label) GetLabel() string {
	if m != nil {
		return m.Label
	}
	return ""
}

func (m *Label) GetScore() float32 {
	if m != nil {
		return m.Score
	}
	return 0
}

type NormalizedCenterRect struct {
	Xc     float32 `protobuf:"fixed32,1,opt,name=xc,proto3" json:"xc,omitempty"`
	Yc     float32 `protobuf:"fixed32,2,opt,name=yc,proto3" json:"yc,omitempty"`
	Width  float32 `protobuf:"fixed32,3,opt,name=width,proto3" json:"width,omitempty"`
	Height float32 `protobuf:"fixed32,4,opt,name=height,proto3" json:"height,omitempty"`
}

func (m *NormalizedCenterRect) Reset()      { *m = NormalizedCenterRect{} }
func (*NormalizedCenterRect) ProtoMessage() {}
func (*NormalizedCenterRect) Descriptor() ([]byte, []int) {
	return fileDescriptor_74a1b3e6342f71eb, []int{1}
}
func (m *NormalizedCenterRect) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *NormalizedCenterRect) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_NormalizedCenterRect.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *NormalizedCenterRect) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NormalizedCenterRect.Merge(m, src)
}
func (m *NormalizedCenterRect) XXX_Size() int {
	return m.Size()
}
func (m *NormalizedCenterRect) XXX_DiscardUnknown() {
	xxx_messageInfo_NormalizedCenterRect.DiscardUnknown(m)
}

var xxx_messageInfo_NormalizedCenterRect proto.InternalMessageInfo

func (m *NormalizedCenterRect) GetXc() float32 {
	if m != nil {
		return m.Xc
	}
	return 0
}

func (m *NormalizedCenterRect) GetYc() float32 {
	if m != nil {
		return m.Yc
	}
	return 0
}

func (m *NormalizedCenterRect) GetWidth() float32 {
	if m != nil {
		return m.Width
	}
	return 0
}

func (m *NormalizedCenterRect) GetHeight() float32 {
	if m != nil {
		return m.Height
	}
	return 0
}

func init() {
	proto.RegisterType((*Label)(nil), "gml.shared.modelout.v1.Label")
	proto.RegisterType((*NormalizedCenterRect)(nil), "gml.shared.modelout.v1.NormalizedCenterRect")
}

func init() {
	proto.RegisterFile("src/shared/modelout/v1/common.proto", fileDescriptor_74a1b3e6342f71eb)
}

var fileDescriptor_74a1b3e6342f71eb = []byte{
	// 270 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x6c, 0x90, 0xb1, 0x4e, 0xc3, 0x30,
	0x14, 0x45, 0x6d, 0x43, 0x2b, 0xe1, 0x81, 0x21, 0xaa, 0xaa, 0x4c, 0x4f, 0x55, 0x59, 0x3a, 0xc5,
	0xaa, 0xca, 0xc6, 0x06, 0x2b, 0x62, 0xc8, 0xc8, 0x96, 0x38, 0x4f, 0x49, 0x90, 0x5d, 0x23, 0xc7,
	0x94, 0x96, 0x89, 0x4f, 0xe0, 0x33, 0xf8, 0x14, 0xc6, 0x8c, 0x1d, 0x89, 0xb3, 0x30, 0xf6, 0x13,
	0x50, 0x92, 0x76, 0xeb, 0xf6, 0xce, 0xd5, 0x7d, 0x67, 0xb8, 0xfc, 0xa6, 0xb2, 0x52, 0x54, 0x45,
	0x62, 0x31, 0x13, 0xda, 0x64, 0xa8, 0xcc, 0x9b, 0x13, 0x9b, 0xa5, 0x90, 0x46, 0x6b, 0xb3, 0x8e,
	0x5e, 0xad, 0x71, 0x26, 0x98, 0xe6, 0x5a, 0x45, 0x43, 0x29, 0x3a, 0x95, 0xa2, 0xcd, 0x72, 0xbe,
	0xe2, 0xa3, 0xc7, 0x24, 0x45, 0x15, 0x4c, 0xf8, 0x48, 0x75, 0x47, 0x48, 0x67, 0x74, 0x71, 0x15,
	0x0f, 0xd0, 0xa5, 0x95, 0x34, 0x16, 0x43, 0x36, 0xa3, 0x0b, 0x16, 0x0f, 0x30, 0xcf, 0xf8, 0xe4,
	0xc9, 0x58, 0x9d, 0xa8, 0xf2, 0x03, 0xb3, 0x07, 0x5c, 0x3b, 0xb4, 0x31, 0x4a, 0x17, 0x5c, 0x73,
	0xb6, 0x95, 0xbd, 0x80, 0xc5, 0x6c, 0x2b, 0x3b, 0xde, 0xc9, 0xe3, 0x2b, 0xdb, 0xc9, 0xce, 0xf6,
	0x5e, 0x66, 0xae, 0x08, 0x2f, 0x06, 0x5b, 0x0f, 0xc1, 0x94, 0x8f, 0x0b, 0x2c, 0xf3, 0xc2, 0x85,
	0x97, 0x7d, 0x7c, 0xa4, 0xfb, 0x97, 0xba, 0x01, 0xb2, 0x6f, 0x80, 0x1c, 0x1a, 0xa0, 0x9f, 0x1e,
	0xe8, 0xb7, 0x07, 0xfa, 0xe3, 0x81, 0xd6, 0x1e, 0xe8, 0xaf, 0x07, 0xfa, 0xe7, 0x81, 0x1c, 0x3c,
	0xd0, 0xaf, 0x16, 0x48, 0xdd, 0x02, 0xd9, 0xb7, 0x40, 0x9e, 0x6f, 0xf3, 0x52, 0x2b, 0x74, 0x2a,
	0x49, 0xab, 0x28, 0x29, 0xc5, 0x40, 0xe2, 0xfc, 0x44, 0x77, 0xa7, 0x3b, 0x1d, 0xf7, 0x2b, 0xad,
	0xfe, 0x03, 0x00, 0x00, 0xff, 0xff, 0x22, 0x41, 0xd8, 0x1d, 0x4c, 0x01, 0x00, 0x00,
}

func (this *Label) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*Label)
	if !ok {
		that2, ok := that.(Label)
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
	if this.Label != that1.Label {
		return false
	}
	if this.Score != that1.Score {
		return false
	}
	return true
}
func (this *NormalizedCenterRect) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*NormalizedCenterRect)
	if !ok {
		that2, ok := that.(NormalizedCenterRect)
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
	if this.Xc != that1.Xc {
		return false
	}
	if this.Yc != that1.Yc {
		return false
	}
	if this.Width != that1.Width {
		return false
	}
	if this.Height != that1.Height {
		return false
	}
	return true
}
func (this *Label) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&modelout.Label{")
	s = append(s, "Label: "+fmt.Sprintf("%#v", this.Label)+",\n")
	s = append(s, "Score: "+fmt.Sprintf("%#v", this.Score)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *NormalizedCenterRect) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 8)
	s = append(s, "&modelout.NormalizedCenterRect{")
	s = append(s, "Xc: "+fmt.Sprintf("%#v", this.Xc)+",\n")
	s = append(s, "Yc: "+fmt.Sprintf("%#v", this.Yc)+",\n")
	s = append(s, "Width: "+fmt.Sprintf("%#v", this.Width)+",\n")
	s = append(s, "Height: "+fmt.Sprintf("%#v", this.Height)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringCommon(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *Label) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Label) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Label) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Score != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Score))))
		i--
		dAtA[i] = 0x15
	}
	if len(m.Label) > 0 {
		i -= len(m.Label)
		copy(dAtA[i:], m.Label)
		i = encodeVarintCommon(dAtA, i, uint64(len(m.Label)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *NormalizedCenterRect) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NormalizedCenterRect) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *NormalizedCenterRect) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Height != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Height))))
		i--
		dAtA[i] = 0x25
	}
	if m.Width != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Width))))
		i--
		dAtA[i] = 0x1d
	}
	if m.Yc != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Yc))))
		i--
		dAtA[i] = 0x15
	}
	if m.Xc != 0 {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Xc))))
		i--
		dAtA[i] = 0xd
	}
	return len(dAtA) - i, nil
}

func encodeVarintCommon(dAtA []byte, offset int, v uint64) int {
	offset -= sovCommon(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *Label) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Label)
	if l > 0 {
		n += 1 + l + sovCommon(uint64(l))
	}
	if m.Score != 0 {
		n += 5
	}
	return n
}

func (m *NormalizedCenterRect) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Xc != 0 {
		n += 5
	}
	if m.Yc != 0 {
		n += 5
	}
	if m.Width != 0 {
		n += 5
	}
	if m.Height != 0 {
		n += 5
	}
	return n
}

func sovCommon(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozCommon(x uint64) (n int) {
	return sovCommon(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *Label) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&Label{`,
		`Label:` + fmt.Sprintf("%v", this.Label) + `,`,
		`Score:` + fmt.Sprintf("%v", this.Score) + `,`,
		`}`,
	}, "")
	return s
}
func (this *NormalizedCenterRect) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&NormalizedCenterRect{`,
		`Xc:` + fmt.Sprintf("%v", this.Xc) + `,`,
		`Yc:` + fmt.Sprintf("%v", this.Yc) + `,`,
		`Width:` + fmt.Sprintf("%v", this.Width) + `,`,
		`Height:` + fmt.Sprintf("%v", this.Height) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringCommon(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *Label) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCommon
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
			return fmt.Errorf("proto: Label: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Label: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Label", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowCommon
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
				return ErrInvalidLengthCommon
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthCommon
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Label = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Score", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Score = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipCommon(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthCommon
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
func (m *NormalizedCenterRect) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowCommon
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
			return fmt.Errorf("proto: NormalizedCenterRect: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: NormalizedCenterRect: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Xc", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Xc = float32(math.Float32frombits(v))
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Yc", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Yc = float32(math.Float32frombits(v))
		case 3:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Width", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Width = float32(math.Float32frombits(v))
		case 4:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Height", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Height = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipCommon(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthCommon
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
func skipCommon(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowCommon
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
					return 0, ErrIntOverflowCommon
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
					return 0, ErrIntOverflowCommon
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
				return 0, ErrInvalidLengthCommon
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupCommon
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthCommon
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthCommon        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowCommon          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupCommon = fmt.Errorf("proto: unexpected end of group")
)
