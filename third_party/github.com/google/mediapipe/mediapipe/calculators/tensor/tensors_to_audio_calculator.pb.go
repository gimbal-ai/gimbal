// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/tensor/tensors_to_audio_calculator.proto

package tensor

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	framework "github.com/google/mediapipe/mediapipe/framework"
	io "io"
	math "math"
	math_bits "math/bits"
	reflect "reflect"
	strconv "strconv"
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

type TensorsToAudioCalculatorOptions_DftTensorFormat int32

const (
	T2A_DFT_TENSOR_FORMAT_UNKNOWN TensorsToAudioCalculatorOptions_DftTensorFormat = 0
	T2A_WITHOUT_DC_AND_NYQUIST    TensorsToAudioCalculatorOptions_DftTensorFormat = 1
	T2A_WITH_NYQUIST              TensorsToAudioCalculatorOptions_DftTensorFormat = 2
	T2A_WITH_DC_AND_NYQUIST       TensorsToAudioCalculatorOptions_DftTensorFormat = 3
)

var TensorsToAudioCalculatorOptions_DftTensorFormat_name = map[int32]string{
	0: "T2A_DFT_TENSOR_FORMAT_UNKNOWN",
	1: "T2A_WITHOUT_DC_AND_NYQUIST",
	2: "T2A_WITH_NYQUIST",
	3: "T2A_WITH_DC_AND_NYQUIST",
}

var TensorsToAudioCalculatorOptions_DftTensorFormat_value = map[string]int32{
	"T2A_DFT_TENSOR_FORMAT_UNKNOWN": 0,
	"T2A_WITHOUT_DC_AND_NYQUIST":    1,
	"T2A_WITH_NYQUIST":              2,
	"T2A_WITH_DC_AND_NYQUIST":       3,
}

func (x TensorsToAudioCalculatorOptions_DftTensorFormat) Enum() *TensorsToAudioCalculatorOptions_DftTensorFormat {
	p := new(TensorsToAudioCalculatorOptions_DftTensorFormat)
	*p = x
	return p
}

func (x TensorsToAudioCalculatorOptions_DftTensorFormat) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(TensorsToAudioCalculatorOptions_DftTensorFormat_name, int32(x))
}

func (x *TensorsToAudioCalculatorOptions_DftTensorFormat) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(TensorsToAudioCalculatorOptions_DftTensorFormat_value, data, "TensorsToAudioCalculatorOptions_DftTensorFormat")
	if err != nil {
		return err
	}
	*x = TensorsToAudioCalculatorOptions_DftTensorFormat(value)
	return nil
}

func (TensorsToAudioCalculatorOptions_DftTensorFormat) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_f4e7d216ca548a33, []int{0, 0}
}

type TensorsToAudioCalculatorOptions struct {
	FftSize               int64                                            `protobuf:"varint,1,opt,name=fft_size,json=fftSize" json:"fft_size"`
	NumSamples            int64                                            `protobuf:"varint,2,opt,name=num_samples,json=numSamples" json:"num_samples"`
	NumOverlappingSamples *int64                                           `protobuf:"varint,3,opt,name=num_overlapping_samples,json=numOverlappingSamples,def=0" json:"num_overlapping_samples,omitempty"`
	DftTensorFormat       *TensorsToAudioCalculatorOptions_DftTensorFormat `protobuf:"varint,11,opt,name=dft_tensor_format,json=dftTensorFormat,enum=mediapipe.TensorsToAudioCalculatorOptions_DftTensorFormat,def=2" json:"dft_tensor_format,omitempty"`
	VolumeGainDb          float64                                          `protobuf:"fixed64,12,opt,name=volume_gain_db,json=volumeGainDb" json:"volume_gain_db"`
}

func (m *TensorsToAudioCalculatorOptions) Reset()      { *m = TensorsToAudioCalculatorOptions{} }
func (*TensorsToAudioCalculatorOptions) ProtoMessage() {}
func (*TensorsToAudioCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_f4e7d216ca548a33, []int{0}
}
func (m *TensorsToAudioCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TensorsToAudioCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TensorsToAudioCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TensorsToAudioCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TensorsToAudioCalculatorOptions.Merge(m, src)
}
func (m *TensorsToAudioCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *TensorsToAudioCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_TensorsToAudioCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_TensorsToAudioCalculatorOptions proto.InternalMessageInfo

const Default_TensorsToAudioCalculatorOptions_NumOverlappingSamples int64 = 0
const Default_TensorsToAudioCalculatorOptions_DftTensorFormat TensorsToAudioCalculatorOptions_DftTensorFormat = T2A_WITH_NYQUIST

func (m *TensorsToAudioCalculatorOptions) GetFftSize() int64 {
	if m != nil {
		return m.FftSize
	}
	return 0
}

func (m *TensorsToAudioCalculatorOptions) GetNumSamples() int64 {
	if m != nil {
		return m.NumSamples
	}
	return 0
}

func (m *TensorsToAudioCalculatorOptions) GetNumOverlappingSamples() int64 {
	if m != nil && m.NumOverlappingSamples != nil {
		return *m.NumOverlappingSamples
	}
	return Default_TensorsToAudioCalculatorOptions_NumOverlappingSamples
}

func (m *TensorsToAudioCalculatorOptions) GetDftTensorFormat() TensorsToAudioCalculatorOptions_DftTensorFormat {
	if m != nil && m.DftTensorFormat != nil {
		return *m.DftTensorFormat
	}
	return Default_TensorsToAudioCalculatorOptions_DftTensorFormat
}

func (m *TensorsToAudioCalculatorOptions) GetVolumeGainDb() float64 {
	if m != nil {
		return m.VolumeGainDb
	}
	return 0
}

var E_TensorsToAudioCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*TensorsToAudioCalculatorOptions)(nil),
	Field:         484297136,
	Name:          "mediapipe.TensorsToAudioCalculatorOptions.ext",
	Tag:           "bytes,484297136,opt,name=ext",
	Filename:      "mediapipe/calculators/tensor/tensors_to_audio_calculator.proto",
}

func init() {
	proto.RegisterEnum("mediapipe.TensorsToAudioCalculatorOptions_DftTensorFormat", TensorsToAudioCalculatorOptions_DftTensorFormat_name, TensorsToAudioCalculatorOptions_DftTensorFormat_value)
	proto.RegisterExtension(E_TensorsToAudioCalculatorOptions_Ext)
	proto.RegisterType((*TensorsToAudioCalculatorOptions)(nil), "mediapipe.TensorsToAudioCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/tensor/tensors_to_audio_calculator.proto", fileDescriptor_f4e7d216ca548a33)
}

var fileDescriptor_f4e7d216ca548a33 = []byte{
	// 492 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x92, 0xb1, 0x6f, 0xd3, 0x40,
	0x14, 0xc6, 0x7d, 0x4d, 0x25, 0xe0, 0x5a, 0xb5, 0xe1, 0x04, 0xaa, 0x55, 0xe0, 0x12, 0x2a, 0x90,
	0xa2, 0x0e, 0x36, 0xca, 0x04, 0x19, 0x90, 0xd2, 0x9a, 0x40, 0x85, 0xb0, 0x85, 0xe3, 0xa8, 0x82,
	0x81, 0xd3, 0x25, 0x3e, 0x9b, 0x13, 0xb6, 0xcf, 0xb2, 0xcf, 0xa5, 0xea, 0xc4, 0xc6, 0x8a, 0xc4,
	0x3f, 0xc1, 0xc8, 0xc0, 0x1f, 0xd1, 0x31, 0x63, 0x27, 0x44, 0x1c, 0x09, 0x31, 0x76, 0x63, 0x45,
	0xc6, 0x8d, 0x53, 0x52, 0x09, 0x98, 0x2c, 0x3d, 0xff, 0xbe, 0xef, 0xbd, 0xf7, 0xdd, 0x83, 0x0f,
	0x43, 0xe6, 0x72, 0x1a, 0xf3, 0x98, 0xe9, 0x23, 0x1a, 0x8c, 0xb2, 0x80, 0x4a, 0x91, 0xa4, 0xba,
	0x64, 0x51, 0x2a, 0x92, 0xb3, 0x4f, 0x4a, 0xa4, 0x20, 0x34, 0x73, 0xb9, 0x20, 0x73, 0x46, 0x8b,
	0x13, 0x21, 0x05, 0xba, 0x52, 0xe9, 0x37, 0xef, 0xcc, 0xad, 0xbc, 0x84, 0x86, 0xec, 0xad, 0x48,
	0xde, 0xe8, 0x8b, 0x82, 0xad, 0x2f, 0xcb, 0xb0, 0xe1, 0x94, 0xb6, 0x8e, 0xe8, 0x16, 0xa6, 0xbb,
	0x15, 0x62, 0xc5, 0x92, 0x8b, 0x28, 0x45, 0x0d, 0x78, 0xd9, 0xf3, 0x24, 0x49, 0xf9, 0x11, 0x53,
	0x41, 0x13, 0xb4, 0x6a, 0x3b, 0xcb, 0xc7, 0x5f, 0x1b, 0x8a, 0x7d, 0xc9, 0xf3, 0x64, 0x9f, 0x1f,
	0x31, 0x74, 0x17, 0xae, 0x44, 0x59, 0x48, 0x52, 0x1a, 0xc6, 0x01, 0x4b, 0xd5, 0xa5, 0x73, 0x0c,
	0x8c, 0xb2, 0xb0, 0x5f, 0xd6, 0xd1, 0x03, 0xb8, 0x51, 0x60, 0xe2, 0x80, 0x25, 0x01, 0x8d, 0x63,
	0x1e, 0xf9, 0x95, 0xa4, 0x56, 0x48, 0x3a, 0xe0, 0x9e, 0x7d, 0x3d, 0xca, 0x42, 0x6b, 0x0e, 0xcc,
	0xa4, 0x87, 0xf0, 0xaa, 0xeb, 0x49, 0x52, 0x06, 0x40, 0x3c, 0x91, 0x84, 0x54, 0xaa, 0x2b, 0x4d,
	0xd0, 0x5a, 0x6b, 0x77, 0xb4, 0x6a, 0x51, 0xed, 0x1f, 0x9b, 0x68, 0x86, 0x27, 0x4b, 0xa4, 0xf7,
	0xdb, 0xa1, 0x53, 0x77, 0xda, 0x5d, 0xb2, 0xbf, 0xe7, 0x3c, 0x21, 0xe6, 0x8b, 0xe7, 0x83, 0xbd,
	0xbe, 0x63, 0xaf, 0xbb, 0x7f, 0x22, 0x68, 0x1b, 0xae, 0x1d, 0x88, 0x20, 0x0b, 0x19, 0xf1, 0x29,
	0x8f, 0x88, 0x3b, 0x54, 0x57, 0x9b, 0xa0, 0x05, 0xce, 0xd6, 0x5b, 0x2d, 0xff, 0x3d, 0xa6, 0x3c,
	0x32, 0x86, 0x5b, 0xef, 0x01, 0x5c, 0x5f, 0x68, 0x81, 0x6e, 0xc3, 0x5b, 0x45, 0x13, 0xa3, 0xe7,
	0x10, 0xe7, 0x91, 0xd9, 0xb7, 0x6c, 0xd2, 0xb3, 0xec, 0x67, 0x5d, 0x87, 0x0c, 0xcc, 0xa7, 0xa6,
	0xb5, 0x6f, 0xd6, 0x15, 0x84, 0xe1, 0xe6, 0x6c, 0x0e, 0x6b, 0xe0, 0x10, 0x63, 0x97, 0x74, 0x4d,
	0x63, 0x36, 0x51, 0x1d, 0xa0, 0x6b, 0xf0, 0xc2, 0x9c, 0xf5, 0x25, 0x74, 0x03, 0x6e, 0x54, 0xd5,
	0x05, 0x49, 0xad, 0xfd, 0x0a, 0xd6, 0xd8, 0xa1, 0x44, 0x37, 0xcf, 0x65, 0x73, 0x21, 0x0d, 0xf5,
	0xf3, 0xc7, 0x9f, 0xdf, 0x8b, 0xf7, 0x5c, 0x69, 0x6f, 0xff, 0x7f, 0x86, 0x76, 0x61, 0xbc, 0x13,
	0x8d, 0x27, 0x58, 0x39, 0x99, 0x60, 0xe5, 0x74, 0x82, 0xc1, 0xbb, 0x1c, 0x83, 0x4f, 0x39, 0x06,
	0xc7, 0x39, 0x06, 0xe3, 0x1c, 0x83, 0x6f, 0x39, 0x06, 0x3f, 0x72, 0xac, 0x9c, 0xe6, 0x18, 0x7c,
	0x98, 0x62, 0x65, 0x3c, 0xc5, 0xca, 0xc9, 0x14, 0x2b, 0x2f, 0xef, 0xfb, 0x5c, 0xbe, 0xce, 0x86,
	0xda, 0x48, 0x84, 0xba, 0x2f, 0x84, 0x1f, 0x30, 0x7d, 0x7e, 0xa4, 0x7f, 0xbb, 0xfc, 0x5f, 0x01,
	0x00, 0x00, 0xff, 0xff, 0xec, 0x0d, 0x3c, 0x44, 0x18, 0x03, 0x00, 0x00,
}

func (x TensorsToAudioCalculatorOptions_DftTensorFormat) String() string {
	s, ok := TensorsToAudioCalculatorOptions_DftTensorFormat_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *TensorsToAudioCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*TensorsToAudioCalculatorOptions)
	if !ok {
		that2, ok := that.(TensorsToAudioCalculatorOptions)
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
	if this.FftSize != that1.FftSize {
		return false
	}
	if this.NumSamples != that1.NumSamples {
		return false
	}
	if this.NumOverlappingSamples != nil && that1.NumOverlappingSamples != nil {
		if *this.NumOverlappingSamples != *that1.NumOverlappingSamples {
			return false
		}
	} else if this.NumOverlappingSamples != nil {
		return false
	} else if that1.NumOverlappingSamples != nil {
		return false
	}
	if this.DftTensorFormat != nil && that1.DftTensorFormat != nil {
		if *this.DftTensorFormat != *that1.DftTensorFormat {
			return false
		}
	} else if this.DftTensorFormat != nil {
		return false
	} else if that1.DftTensorFormat != nil {
		return false
	}
	if this.VolumeGainDb != that1.VolumeGainDb {
		return false
	}
	return true
}
func (this *TensorsToAudioCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 9)
	s = append(s, "&tensor.TensorsToAudioCalculatorOptions{")
	s = append(s, "FftSize: "+fmt.Sprintf("%#v", this.FftSize)+",\n")
	s = append(s, "NumSamples: "+fmt.Sprintf("%#v", this.NumSamples)+",\n")
	if this.NumOverlappingSamples != nil {
		s = append(s, "NumOverlappingSamples: "+valueToGoStringTensorsToAudioCalculator(this.NumOverlappingSamples, "int64")+",\n")
	}
	if this.DftTensorFormat != nil {
		s = append(s, "DftTensorFormat: "+valueToGoStringTensorsToAudioCalculator(this.DftTensorFormat, "TensorsToAudioCalculatorOptions_DftTensorFormat")+",\n")
	}
	s = append(s, "VolumeGainDb: "+fmt.Sprintf("%#v", this.VolumeGainDb)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringTensorsToAudioCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *TensorsToAudioCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TensorsToAudioCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *TensorsToAudioCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= 8
	encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.VolumeGainDb))))
	i--
	dAtA[i] = 0x61
	if m.DftTensorFormat != nil {
		i = encodeVarintTensorsToAudioCalculator(dAtA, i, uint64(*m.DftTensorFormat))
		i--
		dAtA[i] = 0x58
	}
	if m.NumOverlappingSamples != nil {
		i = encodeVarintTensorsToAudioCalculator(dAtA, i, uint64(*m.NumOverlappingSamples))
		i--
		dAtA[i] = 0x18
	}
	i = encodeVarintTensorsToAudioCalculator(dAtA, i, uint64(m.NumSamples))
	i--
	dAtA[i] = 0x10
	i = encodeVarintTensorsToAudioCalculator(dAtA, i, uint64(m.FftSize))
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func encodeVarintTensorsToAudioCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovTensorsToAudioCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *TensorsToAudioCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 1 + sovTensorsToAudioCalculator(uint64(m.FftSize))
	n += 1 + sovTensorsToAudioCalculator(uint64(m.NumSamples))
	if m.NumOverlappingSamples != nil {
		n += 1 + sovTensorsToAudioCalculator(uint64(*m.NumOverlappingSamples))
	}
	if m.DftTensorFormat != nil {
		n += 1 + sovTensorsToAudioCalculator(uint64(*m.DftTensorFormat))
	}
	n += 9
	return n
}

func sovTensorsToAudioCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozTensorsToAudioCalculator(x uint64) (n int) {
	return sovTensorsToAudioCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *TensorsToAudioCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&TensorsToAudioCalculatorOptions{`,
		`FftSize:` + fmt.Sprintf("%v", this.FftSize) + `,`,
		`NumSamples:` + fmt.Sprintf("%v", this.NumSamples) + `,`,
		`NumOverlappingSamples:` + valueToStringTensorsToAudioCalculator(this.NumOverlappingSamples) + `,`,
		`DftTensorFormat:` + valueToStringTensorsToAudioCalculator(this.DftTensorFormat) + `,`,
		`VolumeGainDb:` + fmt.Sprintf("%v", this.VolumeGainDb) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringTensorsToAudioCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *TensorsToAudioCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorsToAudioCalculator
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
			return fmt.Errorf("proto: TensorsToAudioCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TensorsToAudioCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field FftSize", wireType)
			}
			m.FftSize = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorsToAudioCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.FftSize |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NumSamples", wireType)
			}
			m.NumSamples = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorsToAudioCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.NumSamples |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NumOverlappingSamples", wireType)
			}
			var v int64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorsToAudioCalculator
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
			m.NumOverlappingSamples = &v
		case 11:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field DftTensorFormat", wireType)
			}
			var v TensorsToAudioCalculatorOptions_DftTensorFormat
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorsToAudioCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= TensorsToAudioCalculatorOptions_DftTensorFormat(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.DftTensorFormat = &v
		case 12:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field VolumeGainDb", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.VolumeGainDb = float64(math.Float64frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipTensorsToAudioCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTensorsToAudioCalculator
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
func skipTensorsToAudioCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTensorsToAudioCalculator
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
					return 0, ErrIntOverflowTensorsToAudioCalculator
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
					return 0, ErrIntOverflowTensorsToAudioCalculator
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
				return 0, ErrInvalidLengthTensorsToAudioCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupTensorsToAudioCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthTensorsToAudioCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthTensorsToAudioCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTensorsToAudioCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupTensorsToAudioCalculator = fmt.Errorf("proto: unexpected end of group")
)
