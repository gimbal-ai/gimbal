// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/stream_handler.proto

package framework

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

type InputStreamHandlerConfig struct {
	InputStreamHandler *string           `protobuf:"bytes,1,opt,name=input_stream_handler,json=inputStreamHandler,def=DefaultInputStreamHandler" json:"input_stream_handler,omitempty"`
	Options            *MediaPipeOptions `protobuf:"bytes,3,opt,name=options" json:"options,omitempty"`
}

func (m *InputStreamHandlerConfig) Reset()      { *m = InputStreamHandlerConfig{} }
func (*InputStreamHandlerConfig) ProtoMessage() {}
func (*InputStreamHandlerConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_f5119c834c6aeae2, []int{0}
}
func (m *InputStreamHandlerConfig) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *InputStreamHandlerConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_InputStreamHandlerConfig.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *InputStreamHandlerConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_InputStreamHandlerConfig.Merge(m, src)
}
func (m *InputStreamHandlerConfig) XXX_Size() int {
	return m.Size()
}
func (m *InputStreamHandlerConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_InputStreamHandlerConfig.DiscardUnknown(m)
}

var xxx_messageInfo_InputStreamHandlerConfig proto.InternalMessageInfo

const Default_InputStreamHandlerConfig_InputStreamHandler string = "DefaultInputStreamHandler"

func (m *InputStreamHandlerConfig) GetInputStreamHandler() string {
	if m != nil && m.InputStreamHandler != nil {
		return *m.InputStreamHandler
	}
	return Default_InputStreamHandlerConfig_InputStreamHandler
}

func (m *InputStreamHandlerConfig) GetOptions() *MediaPipeOptions {
	if m != nil {
		return m.Options
	}
	return nil
}

type OutputStreamHandlerConfig struct {
	OutputStreamHandler *string           `protobuf:"bytes,1,opt,name=output_stream_handler,json=outputStreamHandler,def=InOrderOutputStreamHandler" json:"output_stream_handler,omitempty"`
	InputSidePacket     []string          `protobuf:"bytes,2,rep,name=input_side_packet,json=inputSidePacket" json:"input_side_packet,omitempty"`
	Options             *MediaPipeOptions `protobuf:"bytes,3,opt,name=options" json:"options,omitempty"`
}

func (m *OutputStreamHandlerConfig) Reset()      { *m = OutputStreamHandlerConfig{} }
func (*OutputStreamHandlerConfig) ProtoMessage() {}
func (*OutputStreamHandlerConfig) Descriptor() ([]byte, []int) {
	return fileDescriptor_f5119c834c6aeae2, []int{1}
}
func (m *OutputStreamHandlerConfig) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *OutputStreamHandlerConfig) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_OutputStreamHandlerConfig.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *OutputStreamHandlerConfig) XXX_Merge(src proto.Message) {
	xxx_messageInfo_OutputStreamHandlerConfig.Merge(m, src)
}
func (m *OutputStreamHandlerConfig) XXX_Size() int {
	return m.Size()
}
func (m *OutputStreamHandlerConfig) XXX_DiscardUnknown() {
	xxx_messageInfo_OutputStreamHandlerConfig.DiscardUnknown(m)
}

var xxx_messageInfo_OutputStreamHandlerConfig proto.InternalMessageInfo

const Default_OutputStreamHandlerConfig_OutputStreamHandler string = "InOrderOutputStreamHandler"

func (m *OutputStreamHandlerConfig) GetOutputStreamHandler() string {
	if m != nil && m.OutputStreamHandler != nil {
		return *m.OutputStreamHandler
	}
	return Default_OutputStreamHandlerConfig_OutputStreamHandler
}

func (m *OutputStreamHandlerConfig) GetInputSidePacket() []string {
	if m != nil {
		return m.InputSidePacket
	}
	return nil
}

func (m *OutputStreamHandlerConfig) GetOptions() *MediaPipeOptions {
	if m != nil {
		return m.Options
	}
	return nil
}

func init() {
	proto.RegisterType((*InputStreamHandlerConfig)(nil), "mediapipe.InputStreamHandlerConfig")
	proto.RegisterType((*OutputStreamHandlerConfig)(nil), "mediapipe.OutputStreamHandlerConfig")
}

func init() {
	proto.RegisterFile("mediapipe/framework/stream_handler.proto", fileDescriptor_f5119c834c6aeae2)
}

var fileDescriptor_f5119c834c6aeae2 = []byte{
	// 342 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x9c, 0x52, 0xbf, 0x4b, 0xfb, 0x40,
	0x14, 0xcf, 0xfb, 0x76, 0xf8, 0xd2, 0x73, 0x10, 0xa3, 0x42, 0x5a, 0xe1, 0x51, 0x3a, 0x05, 0x85,
	0x04, 0x04, 0x17, 0xc7, 0xea, 0x60, 0x11, 0x6d, 0x89, 0x9b, 0x4b, 0x88, 0xcd, 0x35, 0x3d, 0xda,
	0xe4, 0x8e, 0xeb, 0x05, 0x57, 0xff, 0x04, 0x17, 0x57, 0x67, 0xff, 0x14, 0x27, 0xe9, 0xd8, 0xd1,
	0x5e, 0x17, 0xc7, 0xfe, 0x09, 0xd2, 0xa4, 0x56, 0x4b, 0xe2, 0xe2, 0x76, 0x7c, 0xde, 0xfb, 0xfc,
	0xe2, 0x1e, 0xb1, 0x63, 0x1a, 0xb2, 0x40, 0x30, 0x41, 0xdd, 0xbe, 0x0c, 0x62, 0x7a, 0xcf, 0xe5,
	0xd0, 0x1d, 0x2b, 0x49, 0x83, 0xd8, 0x1f, 0x04, 0x49, 0x38, 0xa2, 0xd2, 0x11, 0x92, 0x2b, 0x6e,
	0x56, 0xd7, 0x9b, 0xf5, 0xa3, 0x32, 0xd2, 0x1a, 0xf3, 0xb9, 0x50, 0x8c, 0x27, 0xe3, 0x9c, 0xd7,
	0x7c, 0x06, 0x62, 0xb5, 0x13, 0x91, 0xaa, 0x9b, 0x4c, 0xf5, 0x22, 0x17, 0x3d, 0xe3, 0x49, 0x9f,
	0x45, 0xe6, 0x25, 0xd9, 0x63, 0xcb, 0x99, 0xbf, 0x69, 0x69, 0x41, 0x03, 0xec, 0xea, 0x69, 0xed,
	0x9c, 0xf6, 0x83, 0x74, 0xa4, 0x8a, 0x74, 0xcf, 0x64, 0x05, 0xcc, 0x3c, 0x21, 0xff, 0x57, 0xd6,
	0x56, 0xa5, 0x01, 0xf6, 0xd6, 0xf1, 0x81, 0xb3, 0x0e, 0xe5, 0x5c, 0x2d, 0x5f, 0x5d, 0x26, 0x68,
	0x27, 0x5f, 0xf1, 0xbe, 0x76, 0x9b, 0x6f, 0x40, 0x6a, 0x9d, 0x54, 0xfd, 0x92, 0xf0, 0x9a, 0xec,
	0xf3, 0x6c, 0x58, 0x1e, 0xb1, 0xde, 0x4e, 0x3a, 0x32, 0xa4, 0xb2, 0x44, 0xc0, 0xdb, 0xe5, 0x45,
	0xd0, 0x3c, 0x24, 0x3b, 0xab, 0xc6, 0x2c, 0xa4, 0xbe, 0x08, 0x7a, 0x43, 0xaa, 0xac, 0x7f, 0x8d,
	0x8a, 0x5d, 0xf5, 0xb6, 0xf3, 0x4e, 0x2c, 0xa4, 0xdd, 0x0c, 0xfe, 0x63, 0xa1, 0xd6, 0x13, 0x4c,
	0x66, 0x68, 0x4c, 0x67, 0x68, 0x2c, 0x66, 0x08, 0x0f, 0x1a, 0xe1, 0x45, 0x23, 0xbc, 0x6a, 0x84,
	0x89, 0x46, 0x78, 0xd7, 0x08, 0x1f, 0x1a, 0x8d, 0x85, 0x46, 0x78, 0x9c, 0xa3, 0x31, 0x99, 0xa3,
	0x31, 0x9d, 0xa3, 0x41, 0xea, 0x3d, 0x1e, 0x3b, 0x11, 0xe7, 0xd1, 0x88, 0xfe, 0xb0, 0xc9, 0x3e,
	0xb1, 0x65, 0x6e, 0x94, 0xe8, 0x2e, 0xb1, 0x5b, 0x37, 0x62, 0x6a, 0x90, 0xde, 0x39, 0x3d, 0x1e,
	0xbb, 0x39, 0xed, 0xfb, 0x06, 0xdc, 0x92, 0x0b, 0xf9, 0x0c, 0x00, 0x00, 0xff, 0xff, 0xf0, 0x06,
	0xa3, 0x72, 0x6c, 0x02, 0x00, 0x00,
}

func (this *InputStreamHandlerConfig) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*InputStreamHandlerConfig)
	if !ok {
		that2, ok := that.(InputStreamHandlerConfig)
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
	if this.InputStreamHandler != nil && that1.InputStreamHandler != nil {
		if *this.InputStreamHandler != *that1.InputStreamHandler {
			return false
		}
	} else if this.InputStreamHandler != nil {
		return false
	} else if that1.InputStreamHandler != nil {
		return false
	}
	if !this.Options.Equal(that1.Options) {
		return false
	}
	return true
}
func (this *OutputStreamHandlerConfig) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*OutputStreamHandlerConfig)
	if !ok {
		that2, ok := that.(OutputStreamHandlerConfig)
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
	if this.OutputStreamHandler != nil && that1.OutputStreamHandler != nil {
		if *this.OutputStreamHandler != *that1.OutputStreamHandler {
			return false
		}
	} else if this.OutputStreamHandler != nil {
		return false
	} else if that1.OutputStreamHandler != nil {
		return false
	}
	if len(this.InputSidePacket) != len(that1.InputSidePacket) {
		return false
	}
	for i := range this.InputSidePacket {
		if this.InputSidePacket[i] != that1.InputSidePacket[i] {
			return false
		}
	}
	if !this.Options.Equal(that1.Options) {
		return false
	}
	return true
}
func (this *InputStreamHandlerConfig) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&framework.InputStreamHandlerConfig{")
	if this.InputStreamHandler != nil {
		s = append(s, "InputStreamHandler: "+valueToGoStringStreamHandler(this.InputStreamHandler, "string")+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *OutputStreamHandlerConfig) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&framework.OutputStreamHandlerConfig{")
	if this.OutputStreamHandler != nil {
		s = append(s, "OutputStreamHandler: "+valueToGoStringStreamHandler(this.OutputStreamHandler, "string")+",\n")
	}
	if this.InputSidePacket != nil {
		s = append(s, "InputSidePacket: "+fmt.Sprintf("%#v", this.InputSidePacket)+",\n")
	}
	if this.Options != nil {
		s = append(s, "Options: "+fmt.Sprintf("%#v", this.Options)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringStreamHandler(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *InputStreamHandlerConfig) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *InputStreamHandlerConfig) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *InputStreamHandlerConfig) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Options != nil {
		{
			size, err := m.Options.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintStreamHandler(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x1a
	}
	if m.InputStreamHandler != nil {
		i -= len(*m.InputStreamHandler)
		copy(dAtA[i:], *m.InputStreamHandler)
		i = encodeVarintStreamHandler(dAtA, i, uint64(len(*m.InputStreamHandler)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *OutputStreamHandlerConfig) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *OutputStreamHandlerConfig) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *OutputStreamHandlerConfig) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.Options != nil {
		{
			size, err := m.Options.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintStreamHandler(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x1a
	}
	if len(m.InputSidePacket) > 0 {
		for iNdEx := len(m.InputSidePacket) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.InputSidePacket[iNdEx])
			copy(dAtA[i:], m.InputSidePacket[iNdEx])
			i = encodeVarintStreamHandler(dAtA, i, uint64(len(m.InputSidePacket[iNdEx])))
			i--
			dAtA[i] = 0x12
		}
	}
	if m.OutputStreamHandler != nil {
		i -= len(*m.OutputStreamHandler)
		copy(dAtA[i:], *m.OutputStreamHandler)
		i = encodeVarintStreamHandler(dAtA, i, uint64(len(*m.OutputStreamHandler)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintStreamHandler(dAtA []byte, offset int, v uint64) int {
	offset -= sovStreamHandler(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *InputStreamHandlerConfig) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.InputStreamHandler != nil {
		l = len(*m.InputStreamHandler)
		n += 1 + l + sovStreamHandler(uint64(l))
	}
	if m.Options != nil {
		l = m.Options.Size()
		n += 1 + l + sovStreamHandler(uint64(l))
	}
	return n
}

func (m *OutputStreamHandlerConfig) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.OutputStreamHandler != nil {
		l = len(*m.OutputStreamHandler)
		n += 1 + l + sovStreamHandler(uint64(l))
	}
	if len(m.InputSidePacket) > 0 {
		for _, s := range m.InputSidePacket {
			l = len(s)
			n += 1 + l + sovStreamHandler(uint64(l))
		}
	}
	if m.Options != nil {
		l = m.Options.Size()
		n += 1 + l + sovStreamHandler(uint64(l))
	}
	return n
}

func sovStreamHandler(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozStreamHandler(x uint64) (n int) {
	return sovStreamHandler(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *InputStreamHandlerConfig) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&InputStreamHandlerConfig{`,
		`InputStreamHandler:` + valueToStringStreamHandler(this.InputStreamHandler) + `,`,
		`Options:` + strings.Replace(fmt.Sprintf("%v", this.Options), "MediaPipeOptions", "MediaPipeOptions", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *OutputStreamHandlerConfig) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&OutputStreamHandlerConfig{`,
		`OutputStreamHandler:` + valueToStringStreamHandler(this.OutputStreamHandler) + `,`,
		`InputSidePacket:` + fmt.Sprintf("%v", this.InputSidePacket) + `,`,
		`Options:` + strings.Replace(fmt.Sprintf("%v", this.Options), "MediaPipeOptions", "MediaPipeOptions", 1) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringStreamHandler(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *InputStreamHandlerConfig) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowStreamHandler
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
			return fmt.Errorf("proto: InputStreamHandlerConfig: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: InputStreamHandlerConfig: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field InputStreamHandler", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowStreamHandler
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
				return ErrInvalidLengthStreamHandler
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthStreamHandler
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			s := string(dAtA[iNdEx:postIndex])
			m.InputStreamHandler = &s
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Options", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowStreamHandler
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
				return ErrInvalidLengthStreamHandler
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthStreamHandler
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Options == nil {
				m.Options = &MediaPipeOptions{}
			}
			if err := m.Options.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipStreamHandler(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthStreamHandler
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
func (m *OutputStreamHandlerConfig) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowStreamHandler
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
			return fmt.Errorf("proto: OutputStreamHandlerConfig: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: OutputStreamHandlerConfig: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutputStreamHandler", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowStreamHandler
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
				return ErrInvalidLengthStreamHandler
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthStreamHandler
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			s := string(dAtA[iNdEx:postIndex])
			m.OutputStreamHandler = &s
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field InputSidePacket", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowStreamHandler
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
				return ErrInvalidLengthStreamHandler
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthStreamHandler
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.InputSidePacket = append(m.InputSidePacket, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Options", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowStreamHandler
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
				return ErrInvalidLengthStreamHandler
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthStreamHandler
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.Options == nil {
				m.Options = &MediaPipeOptions{}
			}
			if err := m.Options.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipStreamHandler(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthStreamHandler
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
func skipStreamHandler(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowStreamHandler
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
					return 0, ErrIntOverflowStreamHandler
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
					return 0, ErrIntOverflowStreamHandler
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
				return 0, ErrInvalidLengthStreamHandler
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupStreamHandler
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthStreamHandler
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthStreamHandler        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowStreamHandler          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupStreamHandler = fmt.Errorf("proto: unexpected end of group")
)