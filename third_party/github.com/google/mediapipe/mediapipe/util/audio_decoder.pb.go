// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/util/audio_decoder.proto

package util

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

type AudioStreamOptions struct {
	StreamIndex                *int64 `protobuf:"varint,1,opt,name=stream_index,json=streamIndex,def=0" json:"stream_index,omitempty"`
	AllowMissing               *bool  `protobuf:"varint,2,opt,name=allow_missing,json=allowMissing,def=0" json:"allow_missing,omitempty"`
	IgnoreDecodeFailures       *bool  `protobuf:"varint,3,opt,name=ignore_decode_failures,json=ignoreDecodeFailures,def=0" json:"ignore_decode_failures,omitempty"`
	OutputRegressingTimestamps *bool  `protobuf:"varint,4,opt,name=output_regressing_timestamps,json=outputRegressingTimestamps,def=0" json:"output_regressing_timestamps,omitempty"`
	CorrectPtsForRollover      bool   `protobuf:"varint,5,opt,name=correct_pts_for_rollover,json=correctPtsForRollover" json:"correct_pts_for_rollover"`
}

func (m *AudioStreamOptions) Reset()      { *m = AudioStreamOptions{} }
func (*AudioStreamOptions) ProtoMessage() {}
func (*AudioStreamOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_62567da254a86c7e, []int{0}
}
func (m *AudioStreamOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *AudioStreamOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_AudioStreamOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *AudioStreamOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_AudioStreamOptions.Merge(m, src)
}
func (m *AudioStreamOptions) XXX_Size() int {
	return m.Size()
}
func (m *AudioStreamOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_AudioStreamOptions.DiscardUnknown(m)
}

var xxx_messageInfo_AudioStreamOptions proto.InternalMessageInfo

const Default_AudioStreamOptions_StreamIndex int64 = 0
const Default_AudioStreamOptions_AllowMissing bool = false
const Default_AudioStreamOptions_IgnoreDecodeFailures bool = false
const Default_AudioStreamOptions_OutputRegressingTimestamps bool = false

func (m *AudioStreamOptions) GetStreamIndex() int64 {
	if m != nil && m.StreamIndex != nil {
		return *m.StreamIndex
	}
	return Default_AudioStreamOptions_StreamIndex
}

func (m *AudioStreamOptions) GetAllowMissing() bool {
	if m != nil && m.AllowMissing != nil {
		return *m.AllowMissing
	}
	return Default_AudioStreamOptions_AllowMissing
}

func (m *AudioStreamOptions) GetIgnoreDecodeFailures() bool {
	if m != nil && m.IgnoreDecodeFailures != nil {
		return *m.IgnoreDecodeFailures
	}
	return Default_AudioStreamOptions_IgnoreDecodeFailures
}

func (m *AudioStreamOptions) GetOutputRegressingTimestamps() bool {
	if m != nil && m.OutputRegressingTimestamps != nil {
		return *m.OutputRegressingTimestamps
	}
	return Default_AudioStreamOptions_OutputRegressingTimestamps
}

func (m *AudioStreamOptions) GetCorrectPtsForRollover() bool {
	if m != nil {
		return m.CorrectPtsForRollover
	}
	return false
}

type AudioDecoderOptions struct {
	AudioStream []*AudioStreamOptions `protobuf:"bytes,1,rep,name=audio_stream,json=audioStream" json:"audio_stream,omitempty"`
	StartTime   float64               `protobuf:"fixed64,2,opt,name=start_time,json=startTime" json:"start_time"`
	EndTime     float64               `protobuf:"fixed64,3,opt,name=end_time,json=endTime" json:"end_time"`
}

func (m *AudioDecoderOptions) Reset()      { *m = AudioDecoderOptions{} }
func (*AudioDecoderOptions) ProtoMessage() {}
func (*AudioDecoderOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_62567da254a86c7e, []int{1}
}
func (m *AudioDecoderOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *AudioDecoderOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_AudioDecoderOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *AudioDecoderOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_AudioDecoderOptions.Merge(m, src)
}
func (m *AudioDecoderOptions) XXX_Size() int {
	return m.Size()
}
func (m *AudioDecoderOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_AudioDecoderOptions.DiscardUnknown(m)
}

var xxx_messageInfo_AudioDecoderOptions proto.InternalMessageInfo

func (m *AudioDecoderOptions) GetAudioStream() []*AudioStreamOptions {
	if m != nil {
		return m.AudioStream
	}
	return nil
}

func (m *AudioDecoderOptions) GetStartTime() float64 {
	if m != nil {
		return m.StartTime
	}
	return 0
}

func (m *AudioDecoderOptions) GetEndTime() float64 {
	if m != nil {
		return m.EndTime
	}
	return 0
}

var E_AudioDecoderOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*AudioDecoderOptions)(nil),
	Field:         263370674,
	Name:          "mediapipe.AudioDecoderOptions.ext",
	Tag:           "bytes,263370674,opt,name=ext",
	Filename:      "mediapipe/util/audio_decoder.proto",
}

func init() {
	proto.RegisterType((*AudioStreamOptions)(nil), "mediapipe.AudioStreamOptions")
	proto.RegisterExtension(E_AudioDecoderOptions_Ext)
	proto.RegisterType((*AudioDecoderOptions)(nil), "mediapipe.AudioDecoderOptions")
}

func init() {
	proto.RegisterFile("mediapipe/util/audio_decoder.proto", fileDescriptor_62567da254a86c7e)
}

var fileDescriptor_62567da254a86c7e = []byte{
	// 473 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x64, 0x92, 0xbd, 0x6e, 0xd4, 0x40,
	0x14, 0x85, 0x3d, 0x71, 0x22, 0x92, 0xd9, 0xa5, 0x19, 0x7e, 0x64, 0x45, 0x61, 0xb2, 0x5a, 0x52,
	0xac, 0x52, 0xac, 0xd1, 0x96, 0x41, 0x48, 0x10, 0x50, 0x10, 0x05, 0x02, 0x0c, 0x15, 0x8d, 0x19,
	0xec, 0xbb, 0x66, 0xc4, 0xd8, 0xd7, 0x9a, 0x19, 0x93, 0x34, 0x48, 0x3c, 0x02, 0x8f, 0x81, 0x44,
	0xc7, 0x53, 0x44, 0x54, 0x5b, 0xa6, 0x42, 0xac, 0xb7, 0xa1, 0x63, 0x1f, 0x01, 0xf9, 0x07, 0xef,
	0x06, 0xda, 0x33, 0xdf, 0x9c, 0x39, 0xf7, 0xdc, 0xa1, 0xc3, 0x14, 0x62, 0x29, 0x72, 0x99, 0x83,
	0x5f, 0x58, 0xa9, 0x7c, 0x51, 0xc4, 0x12, 0xc3, 0x18, 0x22, 0x8c, 0x41, 0x8f, 0x73, 0x8d, 0x16,
	0xd9, 0x4e, 0xc7, 0xec, 0x1e, 0xac, 0xf0, 0xa9, 0x16, 0x29, 0x9c, 0xa2, 0x7e, 0xef, 0x47, 0x42,
	0x45, 0x85, 0x12, 0x16, 0xdb, 0x0b, 0xc3, 0xaf, 0x1b, 0x94, 0x3d, 0xa8, 0x8c, 0x5e, 0x5a, 0x0d,
	0x22, 0x7d, 0x96, 0x5b, 0x89, 0x99, 0x61, 0x07, 0xb4, 0x6f, 0x6a, 0x21, 0x94, 0x59, 0x0c, 0x67,
	0x1e, 0x19, 0x90, 0x91, 0x7b, 0x44, 0xee, 0x04, 0xbd, 0x46, 0x7e, 0x52, 0xa9, 0xec, 0x90, 0x5e,
	0x15, 0x4a, 0xe1, 0x69, 0x98, 0x4a, 0x63, 0x64, 0x96, 0x78, 0x1b, 0x03, 0x32, 0xda, 0x3e, 0xda,
	0x9a, 0x0a, 0x65, 0x20, 0xe8, 0xd7, 0x67, 0x4f, 0x9b, 0x23, 0x76, 0x97, 0xde, 0x94, 0x49, 0x86,
	0x1a, 0xda, 0xc4, 0xe1, 0x54, 0x48, 0x55, 0x68, 0x30, 0x9e, 0xbb, 0x7e, 0xe9, 0x7a, 0x03, 0x3d,
	0xaa, 0x99, 0x93, 0x16, 0x61, 0x8f, 0xe9, 0x1e, 0x16, 0x36, 0x2f, 0x6c, 0xa8, 0x21, 0xd1, 0x50,
	0x3b, 0x86, 0x56, 0xa6, 0x60, 0xac, 0x48, 0x73, 0xe3, 0x6d, 0xae, 0x5b, 0xec, 0x36, 0x68, 0xd0,
	0x91, 0xaf, 0x3a, 0x90, 0xdd, 0xa3, 0x5e, 0x84, 0x5a, 0x43, 0x64, 0xc3, 0xdc, 0x9a, 0x70, 0x8a,
	0x3a, 0xd4, 0xa8, 0x14, 0x7e, 0x00, 0xed, 0x6d, 0x55, 0x26, 0xc7, 0x9b, 0xe7, 0x3f, 0xf6, 0x9d,
	0xe0, 0x46, 0x4b, 0x3d, 0xb7, 0xe6, 0x04, 0x75, 0xd0, 0x22, 0xc3, 0x25, 0xa1, 0xd7, 0xea, 0xb6,
	0x9a, 0x7c, 0xfa, 0x6f, 0x5d, 0xf7, 0x69, 0xbf, 0xd9, 0x46, 0xd3, 0x8e, 0x47, 0x06, 0xee, 0xa8,
	0x37, 0xb9, 0x35, 0xee, 0x56, 0x30, 0xfe, 0xbf, 0xe3, 0xa0, 0x27, 0x56, 0x1a, 0xbb, 0x4d, 0xa9,
	0xb1, 0x42, 0xdb, 0x7a, 0xaa, 0xba, 0x47, 0xd2, 0x46, 0xd9, 0xa9, 0xf5, 0x6a, 0x06, 0xb6, 0x4f,
	0xb7, 0x21, 0x8b, 0x1b, 0xc4, 0x5d, 0x43, 0xae, 0x40, 0x16, 0x57, 0xc0, 0xe4, 0x05, 0x75, 0xe1,
	0xcc, 0xb2, 0xbd, 0xb5, 0x87, 0x1f, 0x76, 0x1b, 0x6f, 0xdf, 0xf5, 0xbe, 0xfd, 0xfe, 0xfe, 0x71,
	0x40, 0x46, 0xbd, 0x09, 0xff, 0x37, 0xdf, 0xe5, 0xa9, 0x82, 0xca, 0xeb, 0xf8, 0xcd, 0x6c, 0xce,
	0x9d, 0x8b, 0x39, 0x77, 0x96, 0x73, 0x4e, 0x3e, 0x95, 0x9c, 0x7c, 0x29, 0x39, 0x39, 0x2f, 0x39,
	0x99, 0x95, 0x9c, 0xfc, 0x2c, 0x39, 0xf9, 0x55, 0x72, 0x67, 0x59, 0x72, 0xf2, 0x79, 0xc1, 0x9d,
	0xd9, 0x82, 0x3b, 0x17, 0x0b, 0xee, 0xbc, 0x3e, 0x4c, 0xa4, 0x7d, 0x57, 0xbc, 0x1d, 0x47, 0x98,
	0xfa, 0x09, 0x62, 0xa2, 0xc0, 0x5f, 0x7d, 0xc7, 0xcb, 0xff, 0xf8, 0x4f, 0x00, 0x00, 0x00, 0xff,
	0xff, 0xf4, 0x61, 0x69, 0xe2, 0xd8, 0x02, 0x00, 0x00,
}

func (this *AudioStreamOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*AudioStreamOptions)
	if !ok {
		that2, ok := that.(AudioStreamOptions)
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
	if this.StreamIndex != nil && that1.StreamIndex != nil {
		if *this.StreamIndex != *that1.StreamIndex {
			return false
		}
	} else if this.StreamIndex != nil {
		return false
	} else if that1.StreamIndex != nil {
		return false
	}
	if this.AllowMissing != nil && that1.AllowMissing != nil {
		if *this.AllowMissing != *that1.AllowMissing {
			return false
		}
	} else if this.AllowMissing != nil {
		return false
	} else if that1.AllowMissing != nil {
		return false
	}
	if this.IgnoreDecodeFailures != nil && that1.IgnoreDecodeFailures != nil {
		if *this.IgnoreDecodeFailures != *that1.IgnoreDecodeFailures {
			return false
		}
	} else if this.IgnoreDecodeFailures != nil {
		return false
	} else if that1.IgnoreDecodeFailures != nil {
		return false
	}
	if this.OutputRegressingTimestamps != nil && that1.OutputRegressingTimestamps != nil {
		if *this.OutputRegressingTimestamps != *that1.OutputRegressingTimestamps {
			return false
		}
	} else if this.OutputRegressingTimestamps != nil {
		return false
	} else if that1.OutputRegressingTimestamps != nil {
		return false
	}
	if this.CorrectPtsForRollover != that1.CorrectPtsForRollover {
		return false
	}
	return true
}
func (this *AudioDecoderOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*AudioDecoderOptions)
	if !ok {
		that2, ok := that.(AudioDecoderOptions)
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
	if len(this.AudioStream) != len(that1.AudioStream) {
		return false
	}
	for i := range this.AudioStream {
		if !this.AudioStream[i].Equal(that1.AudioStream[i]) {
			return false
		}
	}
	if this.StartTime != that1.StartTime {
		return false
	}
	if this.EndTime != that1.EndTime {
		return false
	}
	return true
}
func (this *AudioStreamOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 9)
	s = append(s, "&util.AudioStreamOptions{")
	if this.StreamIndex != nil {
		s = append(s, "StreamIndex: "+valueToGoStringAudioDecoder(this.StreamIndex, "int64")+",\n")
	}
	if this.AllowMissing != nil {
		s = append(s, "AllowMissing: "+valueToGoStringAudioDecoder(this.AllowMissing, "bool")+",\n")
	}
	if this.IgnoreDecodeFailures != nil {
		s = append(s, "IgnoreDecodeFailures: "+valueToGoStringAudioDecoder(this.IgnoreDecodeFailures, "bool")+",\n")
	}
	if this.OutputRegressingTimestamps != nil {
		s = append(s, "OutputRegressingTimestamps: "+valueToGoStringAudioDecoder(this.OutputRegressingTimestamps, "bool")+",\n")
	}
	s = append(s, "CorrectPtsForRollover: "+fmt.Sprintf("%#v", this.CorrectPtsForRollover)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *AudioDecoderOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&util.AudioDecoderOptions{")
	if this.AudioStream != nil {
		s = append(s, "AudioStream: "+fmt.Sprintf("%#v", this.AudioStream)+",\n")
	}
	s = append(s, "StartTime: "+fmt.Sprintf("%#v", this.StartTime)+",\n")
	s = append(s, "EndTime: "+fmt.Sprintf("%#v", this.EndTime)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringAudioDecoder(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *AudioStreamOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *AudioStreamOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *AudioStreamOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i--
	if m.CorrectPtsForRollover {
		dAtA[i] = 1
	} else {
		dAtA[i] = 0
	}
	i--
	dAtA[i] = 0x28
	if m.OutputRegressingTimestamps != nil {
		i--
		if *m.OutputRegressingTimestamps {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x20
	}
	if m.IgnoreDecodeFailures != nil {
		i--
		if *m.IgnoreDecodeFailures {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x18
	}
	if m.AllowMissing != nil {
		i--
		if *m.AllowMissing {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x10
	}
	if m.StreamIndex != nil {
		i = encodeVarintAudioDecoder(dAtA, i, uint64(*m.StreamIndex))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func (m *AudioDecoderOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *AudioDecoderOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *AudioDecoderOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= 8
	encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.EndTime))))
	i--
	dAtA[i] = 0x19
	i -= 8
	encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.StartTime))))
	i--
	dAtA[i] = 0x11
	if len(m.AudioStream) > 0 {
		for iNdEx := len(m.AudioStream) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.AudioStream[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintAudioDecoder(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintAudioDecoder(dAtA []byte, offset int, v uint64) int {
	offset -= sovAudioDecoder(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *AudioStreamOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.StreamIndex != nil {
		n += 1 + sovAudioDecoder(uint64(*m.StreamIndex))
	}
	if m.AllowMissing != nil {
		n += 2
	}
	if m.IgnoreDecodeFailures != nil {
		n += 2
	}
	if m.OutputRegressingTimestamps != nil {
		n += 2
	}
	n += 2
	return n
}

func (m *AudioDecoderOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.AudioStream) > 0 {
		for _, e := range m.AudioStream {
			l = e.Size()
			n += 1 + l + sovAudioDecoder(uint64(l))
		}
	}
	n += 9
	n += 9
	return n
}

func sovAudioDecoder(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozAudioDecoder(x uint64) (n int) {
	return sovAudioDecoder(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *AudioStreamOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&AudioStreamOptions{`,
		`StreamIndex:` + valueToStringAudioDecoder(this.StreamIndex) + `,`,
		`AllowMissing:` + valueToStringAudioDecoder(this.AllowMissing) + `,`,
		`IgnoreDecodeFailures:` + valueToStringAudioDecoder(this.IgnoreDecodeFailures) + `,`,
		`OutputRegressingTimestamps:` + valueToStringAudioDecoder(this.OutputRegressingTimestamps) + `,`,
		`CorrectPtsForRollover:` + fmt.Sprintf("%v", this.CorrectPtsForRollover) + `,`,
		`}`,
	}, "")
	return s
}
func (this *AudioDecoderOptions) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForAudioStream := "[]*AudioStreamOptions{"
	for _, f := range this.AudioStream {
		repeatedStringForAudioStream += strings.Replace(f.String(), "AudioStreamOptions", "AudioStreamOptions", 1) + ","
	}
	repeatedStringForAudioStream += "}"
	s := strings.Join([]string{`&AudioDecoderOptions{`,
		`AudioStream:` + repeatedStringForAudioStream + `,`,
		`StartTime:` + fmt.Sprintf("%v", this.StartTime) + `,`,
		`EndTime:` + fmt.Sprintf("%v", this.EndTime) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringAudioDecoder(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *AudioStreamOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowAudioDecoder
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
			return fmt.Errorf("proto: AudioStreamOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: AudioStreamOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field StreamIndex", wireType)
			}
			var v int64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAudioDecoder
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.StreamIndex = &v
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field AllowMissing", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAudioDecoder
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
			b := bool(v != 0)
			m.AllowMissing = &b
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field IgnoreDecodeFailures", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAudioDecoder
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
			b := bool(v != 0)
			m.IgnoreDecodeFailures = &b
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutputRegressingTimestamps", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAudioDecoder
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
			b := bool(v != 0)
			m.OutputRegressingTimestamps = &b
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field CorrectPtsForRollover", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAudioDecoder
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
			m.CorrectPtsForRollover = bool(v != 0)
		default:
			iNdEx = preIndex
			skippy, err := skipAudioDecoder(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthAudioDecoder
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
func (m *AudioDecoderOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowAudioDecoder
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
			return fmt.Errorf("proto: AudioDecoderOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: AudioDecoderOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field AudioStream", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowAudioDecoder
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
				return ErrInvalidLengthAudioDecoder
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthAudioDecoder
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.AudioStream = append(m.AudioStream, &AudioStreamOptions{})
			if err := m.AudioStream[len(m.AudioStream)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field StartTime", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.StartTime = float64(math.Float64frombits(v))
		case 3:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field EndTime", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.EndTime = float64(math.Float64frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipAudioDecoder(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthAudioDecoder
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
func skipAudioDecoder(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowAudioDecoder
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
					return 0, ErrIntOverflowAudioDecoder
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
					return 0, ErrIntOverflowAudioDecoder
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
				return 0, ErrInvalidLengthAudioDecoder
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupAudioDecoder
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthAudioDecoder
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthAudioDecoder        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowAudioDecoder          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupAudioDecoder = fmt.Errorf("proto: unexpected end of group")
)
