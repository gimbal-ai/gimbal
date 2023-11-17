// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/video/video_pre_stream_calculator.proto

package video

import (
	encoding_binary "encoding/binary"
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

type VideoPreStreamCalculatorOptions struct {
	Fps *VideoPreStreamCalculatorOptions_Fps `protobuf:"bytes,1,opt,name=fps" json:"fps,omitempty"`
}

func (m *VideoPreStreamCalculatorOptions) Reset()      { *m = VideoPreStreamCalculatorOptions{} }
func (*VideoPreStreamCalculatorOptions) ProtoMessage() {}
func (*VideoPreStreamCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_72feb25d9723c3f8, []int{0}
}
func (m *VideoPreStreamCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *VideoPreStreamCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_VideoPreStreamCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *VideoPreStreamCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_VideoPreStreamCalculatorOptions.Merge(m, src)
}
func (m *VideoPreStreamCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *VideoPreStreamCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_VideoPreStreamCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_VideoPreStreamCalculatorOptions proto.InternalMessageInfo

func (m *VideoPreStreamCalculatorOptions) GetFps() *VideoPreStreamCalculatorOptions_Fps {
	if m != nil {
		return m.Fps
	}
	return nil
}

var E_VideoPreStreamCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*VideoPreStreamCalculatorOptions)(nil),
	Field:         151386123,
	Name:          "mediapipe.VideoPreStreamCalculatorOptions.ext",
	Tag:           "bytes,151386123,opt,name=ext",
	Filename:      "mediapipe/calculators/video/video_pre_stream_calculator.proto",
}

type VideoPreStreamCalculatorOptions_Fps struct {
	Value float64                                         `protobuf:"fixed64,1,opt,name=value" json:"value"`
	Ratio *VideoPreStreamCalculatorOptions_Fps_Rational32 `protobuf:"bytes,2,opt,name=ratio" json:"ratio,omitempty"`
}

func (m *VideoPreStreamCalculatorOptions_Fps) Reset()      { *m = VideoPreStreamCalculatorOptions_Fps{} }
func (*VideoPreStreamCalculatorOptions_Fps) ProtoMessage() {}
func (*VideoPreStreamCalculatorOptions_Fps) Descriptor() ([]byte, []int) {
	return fileDescriptor_72feb25d9723c3f8, []int{0, 0}
}
func (m *VideoPreStreamCalculatorOptions_Fps) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *VideoPreStreamCalculatorOptions_Fps) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *VideoPreStreamCalculatorOptions_Fps) XXX_Merge(src proto.Message) {
	xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps.Merge(m, src)
}
func (m *VideoPreStreamCalculatorOptions_Fps) XXX_Size() int {
	return m.Size()
}
func (m *VideoPreStreamCalculatorOptions_Fps) XXX_DiscardUnknown() {
	xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps.DiscardUnknown(m)
}

var xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps proto.InternalMessageInfo

func (m *VideoPreStreamCalculatorOptions_Fps) GetValue() float64 {
	if m != nil {
		return m.Value
	}
	return 0
}

func (m *VideoPreStreamCalculatorOptions_Fps) GetRatio() *VideoPreStreamCalculatorOptions_Fps_Rational32 {
	if m != nil {
		return m.Ratio
	}
	return nil
}

type VideoPreStreamCalculatorOptions_Fps_Rational32 struct {
	Numerator   int32 `protobuf:"varint,1,opt,name=numerator" json:"numerator"`
	Denominator int32 `protobuf:"varint,2,opt,name=denominator" json:"denominator"`
}

func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) Reset() {
	*m = VideoPreStreamCalculatorOptions_Fps_Rational32{}
}
func (*VideoPreStreamCalculatorOptions_Fps_Rational32) ProtoMessage() {}
func (*VideoPreStreamCalculatorOptions_Fps_Rational32) Descriptor() ([]byte, []int) {
	return fileDescriptor_72feb25d9723c3f8, []int{0, 0, 0}
}
func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps_Rational32.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) XXX_Merge(src proto.Message) {
	xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps_Rational32.Merge(m, src)
}
func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) XXX_Size() int {
	return m.Size()
}
func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) XXX_DiscardUnknown() {
	xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps_Rational32.DiscardUnknown(m)
}

var xxx_messageInfo_VideoPreStreamCalculatorOptions_Fps_Rational32 proto.InternalMessageInfo

func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) GetNumerator() int32 {
	if m != nil {
		return m.Numerator
	}
	return 0
}

func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) GetDenominator() int32 {
	if m != nil {
		return m.Denominator
	}
	return 0
}

func init() {
	proto.RegisterExtension(E_VideoPreStreamCalculatorOptions_Ext)
	proto.RegisterType((*VideoPreStreamCalculatorOptions)(nil), "mediapipe.VideoPreStreamCalculatorOptions")
	proto.RegisterType((*VideoPreStreamCalculatorOptions_Fps)(nil), "mediapipe.VideoPreStreamCalculatorOptions.Fps")
	proto.RegisterType((*VideoPreStreamCalculatorOptions_Fps_Rational32)(nil), "mediapipe.VideoPreStreamCalculatorOptions.Fps.Rational32")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/video/video_pre_stream_calculator.proto", fileDescriptor_72feb25d9723c3f8)
}

var fileDescriptor_72feb25d9723c3f8 = []byte{
	// 358 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xb2, 0xcd, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0x2f, 0xcb, 0x4c, 0x49, 0xcd, 0x87, 0x90, 0xf1, 0x05, 0x45, 0xa9, 0xf1, 0xc5, 0x25,
	0x45, 0xa9, 0x89, 0xb9, 0xf1, 0x08, 0x25, 0x7a, 0x05, 0x45, 0xf9, 0x25, 0xf9, 0x42, 0x9c, 0x70,
	0xed, 0x52, 0x2a, 0x08, 0x93, 0xd2, 0x8a, 0x12, 0x73, 0x53, 0xcb, 0xf3, 0x8b, 0xb2, 0xf5, 0xd1,
	0x35, 0x28, 0x4d, 0x64, 0xe6, 0x92, 0x0f, 0x03, 0x19, 0x1b, 0x50, 0x94, 0x1a, 0x0c, 0x36, 0xd4,
	0x19, 0xae, 0xc4, 0xbf, 0xa0, 0x24, 0x33, 0x3f, 0xaf, 0x58, 0xc8, 0x81, 0x8b, 0x39, 0xad, 0xa0,
	0x58, 0x82, 0x51, 0x81, 0x51, 0x83, 0xdb, 0x48, 0x4f, 0x0f, 0x6e, 0xae, 0x1e, 0x01, 0x8d, 0x7a,
	0x6e, 0x05, 0xc5, 0x41, 0x20, 0xad, 0x52, 0x67, 0x18, 0xb9, 0x98, 0xdd, 0x0a, 0x8a, 0x85, 0xa4,
	0xb8, 0x58, 0xcb, 0x12, 0x73, 0x4a, 0x53, 0xc1, 0x66, 0x31, 0x3a, 0xb1, 0x9c, 0xb8, 0x27, 0xcf,
	0x10, 0x04, 0x11, 0x12, 0xf2, 0xe7, 0x62, 0x2d, 0x4a, 0x2c, 0xc9, 0xcc, 0x97, 0x60, 0x02, 0xdb,
	0x63, 0x49, 0x9a, 0x3d, 0x7a, 0x41, 0x20, 0xbd, 0x79, 0x89, 0x39, 0xc6, 0x46, 0x41, 0x10, 0x73,
	0xa4, 0x22, 0xb8, 0xb8, 0x10, 0x82, 0x42, 0x4a, 0x5c, 0x9c, 0x79, 0xa5, 0xb9, 0xa9, 0x45, 0x20,
	0x7d, 0x60, 0xeb, 0x59, 0xa1, 0xd6, 0x23, 0x84, 0x85, 0xd4, 0xb8, 0xb8, 0x53, 0x52, 0xf3, 0xf2,
	0x73, 0x33, 0xf3, 0xc0, 0xaa, 0x98, 0x90, 0x54, 0x21, 0x4b, 0x18, 0xc5, 0x72, 0x31, 0xa7, 0x56,
	0x94, 0x08, 0xc9, 0x20, 0x39, 0x11, 0xc3, 0x51, 0x12, 0xdd, 0x1f, 0xa6, 0x7b, 0x80, 0x7d, 0xa2,
	0x45, 0xbc, 0x4f, 0x82, 0x40, 0xe6, 0x3a, 0xe5, 0x5e, 0x78, 0x28, 0xc7, 0x70, 0xe3, 0xa1, 0x1c,
	0xc3, 0x87, 0x87, 0x72, 0x8c, 0x0d, 0x8f, 0xe4, 0x18, 0x57, 0x3c, 0x92, 0x63, 0x3c, 0xf1, 0x48,
	0x8e, 0xf1, 0xc2, 0x23, 0x39, 0xc6, 0x07, 0x8f, 0xe4, 0x18, 0x5f, 0x3c, 0x92, 0x63, 0xf8, 0xf0,
	0x48, 0x8e, 0x71, 0xc2, 0x63, 0x39, 0x86, 0x0b, 0x8f, 0xe5, 0x18, 0x6e, 0x3c, 0x96, 0x63, 0x88,
	0x32, 0x4f, 0xcf, 0x2c, 0xc9, 0x28, 0x4d, 0xd2, 0x4b, 0xce, 0xcf, 0xd5, 0x4f, 0xcf, 0xcf, 0x4f,
	0xcf, 0x49, 0xd5, 0x47, 0xa4, 0x00, 0x3c, 0xa9, 0x0a, 0x10, 0x00, 0x00, 0xff, 0xff, 0x5e, 0x78,
	0x9e, 0xe6, 0x73, 0x02, 0x00, 0x00,
}

func (this *VideoPreStreamCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*VideoPreStreamCalculatorOptions)
	if !ok {
		that2, ok := that.(VideoPreStreamCalculatorOptions)
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
	if !this.Fps.Equal(that1.Fps) {
		return false
	}
	return true
}
func (this *VideoPreStreamCalculatorOptions_Fps) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*VideoPreStreamCalculatorOptions_Fps)
	if !ok {
		that2, ok := that.(VideoPreStreamCalculatorOptions_Fps)
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
	if this.Value != that1.Value {
		return false
	}
	if !this.Ratio.Equal(that1.Ratio) {
		return false
	}
	return true
}
func (this *VideoPreStreamCalculatorOptions_Fps_Rational32) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*VideoPreStreamCalculatorOptions_Fps_Rational32)
	if !ok {
		that2, ok := that.(VideoPreStreamCalculatorOptions_Fps_Rational32)
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
	if this.Numerator != that1.Numerator {
		return false
	}
	if this.Denominator != that1.Denominator {
		return false
	}
	return true
}
func (this *VideoPreStreamCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&video.VideoPreStreamCalculatorOptions{")
	if this.Fps != nil {
		s = append(s, "Fps: "+fmt.Sprintf("%#v", this.Fps)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *VideoPreStreamCalculatorOptions_Fps) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&video.VideoPreStreamCalculatorOptions_Fps{")
	s = append(s, "Value: "+fmt.Sprintf("%#v", this.Value)+",\n")
	if this.Ratio != nil {
		s = append(s, "Ratio: "+fmt.Sprintf("%#v", this.Ratio)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *VideoPreStreamCalculatorOptions_Fps_Rational32) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&video.VideoPreStreamCalculatorOptions_Fps_Rational32{")
	s = append(s, "Numerator: "+fmt.Sprintf("%#v", this.Numerator)+",\n")
	s = append(s, "Denominator: "+fmt.Sprintf("%#v", this.Denominator)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringVideoPreStreamCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *VideoPreStreamCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *VideoPreStreamCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *VideoPreStreamCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Fps != nil {
		{
			size, err := m.Fps.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintVideoPreStreamCalculator(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *VideoPreStreamCalculatorOptions_Fps) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *VideoPreStreamCalculatorOptions_Fps) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *VideoPreStreamCalculatorOptions_Fps) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Ratio != nil {
		{
			size, err := m.Ratio.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintVideoPreStreamCalculator(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	i -= 8
	encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.Value))))
	i--
	dAtA[i] = 0x9
	return len(dAtA) - i, nil
}

func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i = encodeVarintVideoPreStreamCalculator(dAtA, i, uint64(m.Denominator))
	i--
	dAtA[i] = 0x10
	i = encodeVarintVideoPreStreamCalculator(dAtA, i, uint64(m.Numerator))
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func encodeVarintVideoPreStreamCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovVideoPreStreamCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *VideoPreStreamCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.Fps != nil {
		l = m.Fps.Size()
		n += 1 + l + sovVideoPreStreamCalculator(uint64(l))
	}
	return n
}

func (m *VideoPreStreamCalculatorOptions_Fps) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 9
	if m.Ratio != nil {
		l = m.Ratio.Size()
		n += 1 + l + sovVideoPreStreamCalculator(uint64(l))
	}
	return n
}

func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 1 + sovVideoPreStreamCalculator(uint64(m.Numerator))
	n += 1 + sovVideoPreStreamCalculator(uint64(m.Denominator))
	return n
}

func sovVideoPreStreamCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozVideoPreStreamCalculator(x uint64) (n int) {
	return sovVideoPreStreamCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *VideoPreStreamCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&VideoPreStreamCalculatorOptions{`,
		`Fps:` + strings.Replace(fmt.Sprintf("%v", this.Fps), "VideoPreStreamCalculatorOptions_Fps", "VideoPreStreamCalculatorOptions_Fps", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *VideoPreStreamCalculatorOptions_Fps) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&VideoPreStreamCalculatorOptions_Fps{`,
		`Value:` + fmt.Sprintf("%v", this.Value) + `,`,
		`Ratio:` + strings.Replace(fmt.Sprintf("%v", this.Ratio), "VideoPreStreamCalculatorOptions_Fps_Rational32", "VideoPreStreamCalculatorOptions_Fps_Rational32", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *VideoPreStreamCalculatorOptions_Fps_Rational32) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&VideoPreStreamCalculatorOptions_Fps_Rational32{`,
		`Numerator:` + fmt.Sprintf("%v", this.Numerator) + `,`,
		`Denominator:` + fmt.Sprintf("%v", this.Denominator) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringVideoPreStreamCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *VideoPreStreamCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowVideoPreStreamCalculator
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
			return fmt.Errorf("proto: VideoPreStreamCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: VideoPreStreamCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Fps", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVideoPreStreamCalculator
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
				return ErrInvalidLengthVideoPreStreamCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthVideoPreStreamCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Fps == nil {
				m.Fps = &VideoPreStreamCalculatorOptions_Fps{}
			}
			if err := m.Fps.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipVideoPreStreamCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthVideoPreStreamCalculator
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
func (m *VideoPreStreamCalculatorOptions_Fps) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowVideoPreStreamCalculator
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
			return fmt.Errorf("proto: Fps: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Fps: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field Value", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.Value = float64(math.Float64frombits(v))
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Ratio", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVideoPreStreamCalculator
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
				return ErrInvalidLengthVideoPreStreamCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthVideoPreStreamCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Ratio == nil {
				m.Ratio = &VideoPreStreamCalculatorOptions_Fps_Rational32{}
			}
			if err := m.Ratio.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipVideoPreStreamCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthVideoPreStreamCalculator
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
func (m *VideoPreStreamCalculatorOptions_Fps_Rational32) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowVideoPreStreamCalculator
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
			return fmt.Errorf("proto: Rational32: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Rational32: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Numerator", wireType)
			}
			m.Numerator = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVideoPreStreamCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Numerator |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Denominator", wireType)
			}
			m.Denominator = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowVideoPreStreamCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Denominator |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipVideoPreStreamCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthVideoPreStreamCalculator
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
func skipVideoPreStreamCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowVideoPreStreamCalculator
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
					return 0, ErrIntOverflowVideoPreStreamCalculator
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
					return 0, ErrIntOverflowVideoPreStreamCalculator
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
				return 0, ErrInvalidLengthVideoPreStreamCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupVideoPreStreamCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthVideoPreStreamCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthVideoPreStreamCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowVideoPreStreamCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupVideoPreStreamCalculator = fmt.Errorf("proto: unexpected end of group")
)