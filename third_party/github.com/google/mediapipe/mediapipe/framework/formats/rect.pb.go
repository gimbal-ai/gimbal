// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/formats/rect.proto

package formats

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	github_com_gogo_protobuf_proto "github.com/gogo/protobuf/proto"
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

type Rect struct {
	XCenter  int32    `protobuf:"varint,1,req,name=x_center,json=xCenter" json:"x_center"`
	YCenter  int32    `protobuf:"varint,2,req,name=y_center,json=yCenter" json:"y_center"`
	Height   int32    `protobuf:"varint,3,req,name=height" json:"height"`
	Width    int32    `protobuf:"varint,4,req,name=width" json:"width"`
	Rotation *float32 `protobuf:"fixed32,5,opt,name=rotation,def=0" json:"rotation,omitempty"`
	RectId   int64    `protobuf:"varint,6,opt,name=rect_id,json=rectId" json:"rect_id"`
}

func (m *Rect) Reset()      { *m = Rect{} }
func (*Rect) ProtoMessage() {}
func (*Rect) Descriptor() ([]byte, []int) {
	return fileDescriptor_015658d53091f1d8, []int{0}
}
func (m *Rect) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Rect) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Rect.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Rect) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Rect.Merge(m, src)
}
func (m *Rect) XXX_Size() int {
	return m.Size()
}
func (m *Rect) XXX_DiscardUnknown() {
	xxx_messageInfo_Rect.DiscardUnknown(m)
}

var xxx_messageInfo_Rect proto.InternalMessageInfo

const Default_Rect_Rotation float32 = 0

func (m *Rect) GetXCenter() int32 {
	if m != nil {
		return m.XCenter
	}
	return 0
}

func (m *Rect) GetYCenter() int32 {
	if m != nil {
		return m.YCenter
	}
	return 0
}

func (m *Rect) GetHeight() int32 {
	if m != nil {
		return m.Height
	}
	return 0
}

func (m *Rect) GetWidth() int32 {
	if m != nil {
		return m.Width
	}
	return 0
}

func (m *Rect) GetRotation() float32 {
	if m != nil && m.Rotation != nil {
		return *m.Rotation
	}
	return Default_Rect_Rotation
}

func (m *Rect) GetRectId() int64 {
	if m != nil {
		return m.RectId
	}
	return 0
}

type NormalizedRect struct {
	XCenter  float32  `protobuf:"fixed32,1,req,name=x_center,json=xCenter" json:"x_center"`
	YCenter  float32  `protobuf:"fixed32,2,req,name=y_center,json=yCenter" json:"y_center"`
	Height   float32  `protobuf:"fixed32,3,req,name=height" json:"height"`
	Width    float32  `protobuf:"fixed32,4,req,name=width" json:"width"`
	Rotation *float32 `protobuf:"fixed32,5,opt,name=rotation,def=0" json:"rotation,omitempty"`
	RectId   int64    `protobuf:"varint,6,opt,name=rect_id,json=rectId" json:"rect_id"`
}

func (m *NormalizedRect) Reset()      { *m = NormalizedRect{} }
func (*NormalizedRect) ProtoMessage() {}
func (*NormalizedRect) Descriptor() ([]byte, []int) {
	return fileDescriptor_015658d53091f1d8, []int{1}
}
func (m *NormalizedRect) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *NormalizedRect) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_NormalizedRect.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *NormalizedRect) XXX_Merge(src proto.Message) {
	xxx_messageInfo_NormalizedRect.Merge(m, src)
}
func (m *NormalizedRect) XXX_Size() int {
	return m.Size()
}
func (m *NormalizedRect) XXX_DiscardUnknown() {
	xxx_messageInfo_NormalizedRect.DiscardUnknown(m)
}

var xxx_messageInfo_NormalizedRect proto.InternalMessageInfo

const Default_NormalizedRect_Rotation float32 = 0

func (m *NormalizedRect) GetXCenter() float32 {
	if m != nil {
		return m.XCenter
	}
	return 0
}

func (m *NormalizedRect) GetYCenter() float32 {
	if m != nil {
		return m.YCenter
	}
	return 0
}

func (m *NormalizedRect) GetHeight() float32 {
	if m != nil {
		return m.Height
	}
	return 0
}

func (m *NormalizedRect) GetWidth() float32 {
	if m != nil {
		return m.Width
	}
	return 0
}

func (m *NormalizedRect) GetRotation() float32 {
	if m != nil && m.Rotation != nil {
		return *m.Rotation
	}
	return Default_NormalizedRect_Rotation
}

func (m *NormalizedRect) GetRectId() int64 {
	if m != nil {
		return m.RectId
	}
	return 0
}

func init() {
	proto.RegisterType((*Rect)(nil), "mediapipe.Rect")
	proto.RegisterType((*NormalizedRect)(nil), "mediapipe.NormalizedRect")
}

func init() {
	proto.RegisterFile("mediapipe/framework/formats/rect.proto", fileDescriptor_015658d53091f1d8)
}

var fileDescriptor_015658d53091f1d8 = []byte{
	// 330 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xac, 0x92, 0xbb, 0x4e, 0xc3, 0x30,
	0x14, 0x86, 0x7d, 0xdc, 0xbb, 0x07, 0x86, 0x4c, 0x11, 0xa2, 0xa7, 0x55, 0x07, 0xd4, 0x29, 0x61,
	0x43, 0x62, 0x2c, 0x13, 0x0b, 0x42, 0x1d, 0x59, 0xaa, 0x90, 0xb8, 0x89, 0x45, 0x53, 0x57, 0xc6,
	0xa8, 0x2d, 0x13, 0x8f, 0xc0, 0x03, 0xf0, 0x00, 0x3c, 0x06, 0x63, 0xc5, 0xd4, 0xb1, 0x13, 0xa2,
	0xee, 0xc2, 0xd8, 0x47, 0x40, 0xa6, 0x17, 0x32, 0xa0, 0xb0, 0xb0, 0x45, 0xff, 0xf9, 0x62, 0xf9,
	0xf3, 0x7f, 0xd8, 0x71, 0xca, 0x23, 0x11, 0x8c, 0xc4, 0x88, 0xfb, 0x7d, 0x15, 0xa4, 0x7c, 0x2c,
	0xd5, 0xad, 0xdf, 0x97, 0x2a, 0x0d, 0xf4, 0x9d, 0xaf, 0x78, 0xa8, 0xbd, 0x91, 0x92, 0x5a, 0x3a,
	0xb5, 0x3d, 0xd7, 0x7a, 0x05, 0x56, 0xec, 0xf2, 0x50, 0x3b, 0x0d, 0x56, 0x9d, 0xf4, 0x42, 0x3e,
	0xd4, 0x5c, 0xb9, 0xd0, 0xa4, 0xed, 0x52, 0xa7, 0x38, 0x7b, 0x6f, 0x90, 0x6e, 0x65, 0x72, 0xfe,
	0x1d, 0x5a, 0x60, 0xba, 0x03, 0x68, 0x16, 0x98, 0x6e, 0x81, 0x23, 0x56, 0x4e, 0xb8, 0x88, 0x13,
	0xed, 0x16, 0x32, 0xe3, 0x6d, 0xe6, 0x1c, 0xb2, 0xd2, 0x58, 0x44, 0x3a, 0x71, 0x8b, 0x99, 0xe1,
	0x26, 0x72, 0xea, 0xac, 0xaa, 0xa4, 0x0e, 0xb4, 0x90, 0x43, 0xb7, 0xd4, 0x84, 0x36, 0x3d, 0x83,
	0x93, 0xee, 0x3e, 0x72, 0xea, 0xac, 0x62, 0x2f, 0xdf, 0x13, 0x91, 0x5b, 0x6e, 0x42, 0xbb, 0xb0,
	0x3b, 0xd9, 0x86, 0x17, 0x51, 0xeb, 0x0d, 0xd8, 0xc1, 0xa5, 0x95, 0x1c, 0x88, 0x07, 0x1e, 0xfd,
	0x2a, 0x43, 0xff, 0x92, 0xa1, 0xf9, 0x32, 0x34, 0x4f, 0x86, 0xfe, 0xa3, 0x4c, 0xe7, 0x19, 0xe6,
	0x4b, 0x24, 0x8b, 0x25, 0x92, 0xf5, 0x12, 0xe1, 0xd1, 0x20, 0xbc, 0x18, 0x84, 0x99, 0x41, 0x98,
	0x1b, 0x84, 0x0f, 0x83, 0xf0, 0x69, 0x90, 0xac, 0x0d, 0xc2, 0xd3, 0x0a, 0xc9, 0x7c, 0x85, 0x64,
	0xb1, 0x42, 0xc2, 0x5a, 0xa1, 0x4c, 0xbd, 0x58, 0xca, 0x78, 0xc0, 0xbd, 0x7d, 0xb9, 0xde, 0xb6,
	0xfa, 0x4d, 0xeb, 0x9d, 0x9a, 0x7d, 0x9a, 0x2b, 0xfb, 0x79, 0x7d, 0x1a, 0x0b, 0x9d, 0xdc, 0xdf,
	0x78, 0xa1, 0x4c, 0xfd, 0xcd, 0x5f, 0xfe, 0xcf, 0xea, 0xe4, 0x2c, 0xd1, 0x57, 0x00, 0x00, 0x00,
	0xff, 0xff, 0x18, 0x16, 0x1b, 0x19, 0x62, 0x02, 0x00, 0x00,
}

func (this *Rect) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*Rect)
	if !ok {
		that2, ok := that.(Rect)
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
	if this.XCenter != that1.XCenter {
		return false
	}
	if this.YCenter != that1.YCenter {
		return false
	}
	if this.Height != that1.Height {
		return false
	}
	if this.Width != that1.Width {
		return false
	}
	if this.Rotation != nil && that1.Rotation != nil {
		if *this.Rotation != *that1.Rotation {
			return false
		}
	} else if this.Rotation != nil {
		return false
	} else if that1.Rotation != nil {
		return false
	}
	if this.RectId != that1.RectId {
		return false
	}
	return true
}
func (this *NormalizedRect) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*NormalizedRect)
	if !ok {
		that2, ok := that.(NormalizedRect)
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
	if this.XCenter != that1.XCenter {
		return false
	}
	if this.YCenter != that1.YCenter {
		return false
	}
	if this.Height != that1.Height {
		return false
	}
	if this.Width != that1.Width {
		return false
	}
	if this.Rotation != nil && that1.Rotation != nil {
		if *this.Rotation != *that1.Rotation {
			return false
		}
	} else if this.Rotation != nil {
		return false
	} else if that1.Rotation != nil {
		return false
	}
	if this.RectId != that1.RectId {
		return false
	}
	return true
}
func (this *Rect) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 10)
	s = append(s, "&formats.Rect{")
	s = append(s, "XCenter: "+fmt.Sprintf("%#v", this.XCenter)+",\n")
	s = append(s, "YCenter: "+fmt.Sprintf("%#v", this.YCenter)+",\n")
	s = append(s, "Height: "+fmt.Sprintf("%#v", this.Height)+",\n")
	s = append(s, "Width: "+fmt.Sprintf("%#v", this.Width)+",\n")
	if this.Rotation != nil {
		s = append(s, "Rotation: "+valueToGoStringRect(this.Rotation, "float32")+",\n")
	}
	s = append(s, "RectId: "+fmt.Sprintf("%#v", this.RectId)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *NormalizedRect) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 10)
	s = append(s, "&formats.NormalizedRect{")
	s = append(s, "XCenter: "+fmt.Sprintf("%#v", this.XCenter)+",\n")
	s = append(s, "YCenter: "+fmt.Sprintf("%#v", this.YCenter)+",\n")
	s = append(s, "Height: "+fmt.Sprintf("%#v", this.Height)+",\n")
	s = append(s, "Width: "+fmt.Sprintf("%#v", this.Width)+",\n")
	if this.Rotation != nil {
		s = append(s, "Rotation: "+valueToGoStringRect(this.Rotation, "float32")+",\n")
	}
	s = append(s, "RectId: "+fmt.Sprintf("%#v", this.RectId)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringRect(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *Rect) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Rect) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Rect) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i = encodeVarintRect(dAtA, i, uint64(m.RectId))
	i--
	dAtA[i] = 0x30
	if m.Rotation != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.Rotation))))
		i--
		dAtA[i] = 0x2d
	}
	i = encodeVarintRect(dAtA, i, uint64(m.Width))
	i--
	dAtA[i] = 0x20
	i = encodeVarintRect(dAtA, i, uint64(m.Height))
	i--
	dAtA[i] = 0x18
	i = encodeVarintRect(dAtA, i, uint64(m.YCenter))
	i--
	dAtA[i] = 0x10
	i = encodeVarintRect(dAtA, i, uint64(m.XCenter))
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func (m *NormalizedRect) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *NormalizedRect) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *NormalizedRect) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i = encodeVarintRect(dAtA, i, uint64(m.RectId))
	i--
	dAtA[i] = 0x30
	if m.Rotation != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.Rotation))))
		i--
		dAtA[i] = 0x2d
	}
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Width))))
	i--
	dAtA[i] = 0x25
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Height))))
	i--
	dAtA[i] = 0x1d
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.YCenter))))
	i--
	dAtA[i] = 0x15
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.XCenter))))
	i--
	dAtA[i] = 0xd
	return len(dAtA) - i, nil
}

func encodeVarintRect(dAtA []byte, offset int, v uint64) int {
	offset -= sovRect(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *Rect) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 1 + sovRect(uint64(m.XCenter))
	n += 1 + sovRect(uint64(m.YCenter))
	n += 1 + sovRect(uint64(m.Height))
	n += 1 + sovRect(uint64(m.Width))
	if m.Rotation != nil {
		n += 5
	}
	n += 1 + sovRect(uint64(m.RectId))
	return n
}

func (m *NormalizedRect) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 5
	n += 5
	n += 5
	n += 5
	if m.Rotation != nil {
		n += 5
	}
	n += 1 + sovRect(uint64(m.RectId))
	return n
}

func sovRect(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozRect(x uint64) (n int) {
	return sovRect(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *Rect) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&Rect{`,
		`XCenter:` + fmt.Sprintf("%v", this.XCenter) + `,`,
		`YCenter:` + fmt.Sprintf("%v", this.YCenter) + `,`,
		`Height:` + fmt.Sprintf("%v", this.Height) + `,`,
		`Width:` + fmt.Sprintf("%v", this.Width) + `,`,
		`Rotation:` + valueToStringRect(this.Rotation) + `,`,
		`RectId:` + fmt.Sprintf("%v", this.RectId) + `,`,
		`}`,
	}, "")
	return s
}
func (this *NormalizedRect) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&NormalizedRect{`,
		`XCenter:` + fmt.Sprintf("%v", this.XCenter) + `,`,
		`YCenter:` + fmt.Sprintf("%v", this.YCenter) + `,`,
		`Height:` + fmt.Sprintf("%v", this.Height) + `,`,
		`Width:` + fmt.Sprintf("%v", this.Width) + `,`,
		`Rotation:` + valueToStringRect(this.Rotation) + `,`,
		`RectId:` + fmt.Sprintf("%v", this.RectId) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringRect(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *Rect) Unmarshal(dAtA []byte) error {
	var hasFields [1]uint64
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowRect
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
			return fmt.Errorf("proto: Rect: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Rect: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field XCenter", wireType)
			}
			m.XCenter = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRect
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.XCenter |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			hasFields[0] |= uint64(0x00000001)
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field YCenter", wireType)
			}
			m.YCenter = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRect
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.YCenter |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			hasFields[0] |= uint64(0x00000002)
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Height", wireType)
			}
			m.Height = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRect
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Height |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			hasFields[0] |= uint64(0x00000004)
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Width", wireType)
			}
			m.Width = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRect
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Width |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			hasFields[0] |= uint64(0x00000008)
		case 5:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Rotation", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.Rotation = &v2
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RectId", wireType)
			}
			m.RectId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRect
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.RectId |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipRect(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthRect
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}
	if hasFields[0]&uint64(0x00000001) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("x_center")
	}
	if hasFields[0]&uint64(0x00000002) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("y_center")
	}
	if hasFields[0]&uint64(0x00000004) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("height")
	}
	if hasFields[0]&uint64(0x00000008) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("width")
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func (m *NormalizedRect) Unmarshal(dAtA []byte) error {
	var hasFields [1]uint64
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowRect
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
			return fmt.Errorf("proto: NormalizedRect: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: NormalizedRect: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field XCenter", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.XCenter = float32(math.Float32frombits(v))
			hasFields[0] |= uint64(0x00000001)
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field YCenter", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.YCenter = float32(math.Float32frombits(v))
			hasFields[0] |= uint64(0x00000002)
		case 3:
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
			hasFields[0] |= uint64(0x00000004)
		case 4:
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
			hasFields[0] |= uint64(0x00000008)
		case 5:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Rotation", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.Rotation = &v2
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RectId", wireType)
			}
			m.RectId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowRect
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.RectId |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipRect(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthRect
			}
			if (iNdEx + skippy) > l {
				return io.ErrUnexpectedEOF
			}
			iNdEx += skippy
		}
	}
	if hasFields[0]&uint64(0x00000001) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("x_center")
	}
	if hasFields[0]&uint64(0x00000002) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("y_center")
	}
	if hasFields[0]&uint64(0x00000004) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("height")
	}
	if hasFields[0]&uint64(0x00000008) == 0 {
		return github_com_gogo_protobuf_proto.NewRequiredNotSetError("width")
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipRect(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowRect
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
					return 0, ErrIntOverflowRect
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
					return 0, ErrIntOverflowRect
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
				return 0, ErrInvalidLengthRect
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupRect
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthRect
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthRect        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowRect          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupRect = fmt.Errorf("proto: unexpected end of group")
)