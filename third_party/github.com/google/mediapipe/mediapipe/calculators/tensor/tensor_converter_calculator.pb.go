// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/tensor/tensor_converter_calculator.proto

package tensor

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	framework "github.com/google/mediapipe/mediapipe/framework"
	gpu "github.com/google/mediapipe/mediapipe/gpu"
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

type TensorConverterCalculatorOptions struct {
	ZeroCenter             *bool                                              `protobuf:"varint,1,opt,name=zero_center,json=zeroCenter,def=1" json:"zero_center,omitempty"`
	UseCustomNormalization *bool                                              `protobuf:"varint,6,opt,name=use_custom_normalization,json=useCustomNormalization,def=0" json:"use_custom_normalization,omitempty"`
	CustomDiv              *float32                                           `protobuf:"fixed32,7,opt,name=custom_div,json=customDiv,def=-1" json:"custom_div,omitempty"`
	CustomSub              *float32                                           `protobuf:"fixed32,8,opt,name=custom_sub,json=customSub,def=-1" json:"custom_sub,omitempty"`
	FlipVertically         *bool                                              `protobuf:"varint,2,opt,name=flip_vertically,json=flipVertically,def=0" json:"flip_vertically,omitempty"`
	GpuOrigin              gpu.GpuOrigin_Mode                                 `protobuf:"varint,10,opt,name=gpu_origin,json=gpuOrigin,enum=mediapipe.GpuOrigin_Mode" json:"gpu_origin"`
	MaxNumChannels         *int32                                             `protobuf:"varint,3,opt,name=max_num_channels,json=maxNumChannels,def=3" json:"max_num_channels,omitempty"`
	RowMajorMatrix         *bool                                              `protobuf:"varint,4,opt,name=row_major_matrix,json=rowMajorMatrix,def=0" json:"row_major_matrix,omitempty"`
	UseQuantizedTensors    *bool                                              `protobuf:"varint,5,opt,name=use_quantized_tensors,json=useQuantizedTensors,def=0" json:"use_quantized_tensors,omitempty"`
	OutputTensorFloatRange *TensorConverterCalculatorOptions_TensorFloatRange `protobuf:"bytes,9,opt,name=output_tensor_float_range,json=outputTensorFloatRange" json:"output_tensor_float_range,omitempty"`
}

func (m *TensorConverterCalculatorOptions) Reset()      { *m = TensorConverterCalculatorOptions{} }
func (*TensorConverterCalculatorOptions) ProtoMessage() {}
func (*TensorConverterCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_77bb2b02c0446e43, []int{0}
}
func (m *TensorConverterCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TensorConverterCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TensorConverterCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TensorConverterCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TensorConverterCalculatorOptions.Merge(m, src)
}
func (m *TensorConverterCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *TensorConverterCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_TensorConverterCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_TensorConverterCalculatorOptions proto.InternalMessageInfo

const Default_TensorConverterCalculatorOptions_ZeroCenter bool = true
const Default_TensorConverterCalculatorOptions_UseCustomNormalization bool = false
const Default_TensorConverterCalculatorOptions_CustomDiv float32 = -1
const Default_TensorConverterCalculatorOptions_CustomSub float32 = -1
const Default_TensorConverterCalculatorOptions_FlipVertically bool = false
const Default_TensorConverterCalculatorOptions_MaxNumChannels int32 = 3
const Default_TensorConverterCalculatorOptions_RowMajorMatrix bool = false
const Default_TensorConverterCalculatorOptions_UseQuantizedTensors bool = false

func (m *TensorConverterCalculatorOptions) GetZeroCenter() bool {
	if m != nil && m.ZeroCenter != nil {
		return *m.ZeroCenter
	}
	return Default_TensorConverterCalculatorOptions_ZeroCenter
}

func (m *TensorConverterCalculatorOptions) GetUseCustomNormalization() bool {
	if m != nil && m.UseCustomNormalization != nil {
		return *m.UseCustomNormalization
	}
	return Default_TensorConverterCalculatorOptions_UseCustomNormalization
}

func (m *TensorConverterCalculatorOptions) GetCustomDiv() float32 {
	if m != nil && m.CustomDiv != nil {
		return *m.CustomDiv
	}
	return Default_TensorConverterCalculatorOptions_CustomDiv
}

func (m *TensorConverterCalculatorOptions) GetCustomSub() float32 {
	if m != nil && m.CustomSub != nil {
		return *m.CustomSub
	}
	return Default_TensorConverterCalculatorOptions_CustomSub
}

func (m *TensorConverterCalculatorOptions) GetFlipVertically() bool {
	if m != nil && m.FlipVertically != nil {
		return *m.FlipVertically
	}
	return Default_TensorConverterCalculatorOptions_FlipVertically
}

func (m *TensorConverterCalculatorOptions) GetGpuOrigin() gpu.GpuOrigin_Mode {
	if m != nil {
		return m.GpuOrigin
	}
	return gpu.ORIGIN_MODE_DEFAULT
}

func (m *TensorConverterCalculatorOptions) GetMaxNumChannels() int32 {
	if m != nil && m.MaxNumChannels != nil {
		return *m.MaxNumChannels
	}
	return Default_TensorConverterCalculatorOptions_MaxNumChannels
}

func (m *TensorConverterCalculatorOptions) GetRowMajorMatrix() bool {
	if m != nil && m.RowMajorMatrix != nil {
		return *m.RowMajorMatrix
	}
	return Default_TensorConverterCalculatorOptions_RowMajorMatrix
}

func (m *TensorConverterCalculatorOptions) GetUseQuantizedTensors() bool {
	if m != nil && m.UseQuantizedTensors != nil {
		return *m.UseQuantizedTensors
	}
	return Default_TensorConverterCalculatorOptions_UseQuantizedTensors
}

func (m *TensorConverterCalculatorOptions) GetOutputTensorFloatRange() *TensorConverterCalculatorOptions_TensorFloatRange {
	if m != nil {
		return m.OutputTensorFloatRange
	}
	return nil
}

var E_TensorConverterCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*TensorConverterCalculatorOptions)(nil),
	Field:         335742637,
	Name:          "mediapipe.TensorConverterCalculatorOptions.ext",
	Tag:           "bytes,335742637,opt,name=ext",
	Filename:      "mediapipe/calculators/tensor/tensor_converter_calculator.proto",
}

type TensorConverterCalculatorOptions_TensorFloatRange struct {
	Min float32 `protobuf:"fixed32,1,opt,name=min" json:"min"`
	Max float32 `protobuf:"fixed32,2,opt,name=max" json:"max"`
}

func (m *TensorConverterCalculatorOptions_TensorFloatRange) Reset() {
	*m = TensorConverterCalculatorOptions_TensorFloatRange{}
}
func (*TensorConverterCalculatorOptions_TensorFloatRange) ProtoMessage() {}
func (*TensorConverterCalculatorOptions_TensorFloatRange) Descriptor() ([]byte, []int) {
	return fileDescriptor_77bb2b02c0446e43, []int{0, 0}
}
func (m *TensorConverterCalculatorOptions_TensorFloatRange) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TensorConverterCalculatorOptions_TensorFloatRange) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TensorConverterCalculatorOptions_TensorFloatRange.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TensorConverterCalculatorOptions_TensorFloatRange) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TensorConverterCalculatorOptions_TensorFloatRange.Merge(m, src)
}
func (m *TensorConverterCalculatorOptions_TensorFloatRange) XXX_Size() int {
	return m.Size()
}
func (m *TensorConverterCalculatorOptions_TensorFloatRange) XXX_DiscardUnknown() {
	xxx_messageInfo_TensorConverterCalculatorOptions_TensorFloatRange.DiscardUnknown(m)
}

var xxx_messageInfo_TensorConverterCalculatorOptions_TensorFloatRange proto.InternalMessageInfo

func (m *TensorConverterCalculatorOptions_TensorFloatRange) GetMin() float32 {
	if m != nil {
		return m.Min
	}
	return 0
}

func (m *TensorConverterCalculatorOptions_TensorFloatRange) GetMax() float32 {
	if m != nil {
		return m.Max
	}
	return 0
}

func init() {
	proto.RegisterExtension(E_TensorConverterCalculatorOptions_Ext)
	proto.RegisterType((*TensorConverterCalculatorOptions)(nil), "mediapipe.TensorConverterCalculatorOptions")
	proto.RegisterType((*TensorConverterCalculatorOptions_TensorFloatRange)(nil), "mediapipe.TensorConverterCalculatorOptions.TensorFloatRange")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/tensor/tensor_converter_calculator.proto", fileDescriptor_77bb2b02c0446e43)
}

var fileDescriptor_77bb2b02c0446e43 = []byte{
	// 581 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x53, 0x4d, 0x6b, 0x14, 0x3f,
	0x18, 0x9f, 0x6c, 0xbb, 0xff, 0x7f, 0x37, 0x85, 0x5a, 0x22, 0x96, 0xb4, 0x48, 0x5c, 0x45, 0x61,
	0xa1, 0xb8, 0x8b, 0xf5, 0xa2, 0x45, 0x2a, 0x74, 0x45, 0x4f, 0x6d, 0x71, 0x14, 0x0f, 0x5e, 0x42,
	0x3a, 0x9b, 0x9d, 0x46, 0x27, 0xc9, 0x98, 0x97, 0xee, 0xda, 0x93, 0x1f, 0x40, 0xc1, 0x8f, 0xe0,
	0xd1, 0x8b, 0xdf, 0xa3, 0xc7, 0x1e, 0x7b, 0x12, 0x3b, 0xbd, 0x78, 0xec, 0xcd, 0xab, 0xcc, 0xec,
	0xcb, 0x8c, 0x2b, 0x28, 0x1e, 0x86, 0x81, 0xdf, 0x5b, 0x9e, 0xe7, 0xc9, 0x13, 0xb8, 0x25, 0x79,
	0x4f, 0xb0, 0x54, 0xa4, 0xbc, 0x13, 0xb1, 0x24, 0xf2, 0x09, 0x73, 0xda, 0xd8, 0x8e, 0xe3, 0xca,
	0x6a, 0x33, 0xfe, 0xd1, 0x48, 0xab, 0x43, 0x6e, 0x1c, 0x37, 0xb4, 0xd4, 0xb4, 0x53, 0xa3, 0x9d,
	0x46, 0x8d, 0xa9, 0x7f, 0xed, 0x66, 0x19, 0xd5, 0x37, 0x4c, 0xf2, 0x81, 0x36, 0xaf, 0x3b, 0xb3,
	0x86, 0x35, 0x52, 0xaa, 0xe2, 0xd4, 0xe7, 0x1f, 0xd5, 0x46, 0xc4, 0x42, 0x8d, 0xf8, 0x1b, 0x3f,
	0xea, 0xb0, 0xf9, 0xbc, 0x38, 0xb6, 0x3b, 0x39, 0xb5, 0x3b, 0xcd, 0xd8, 0x4b, 0x9d, 0xd0, 0xca,
	0xa2, 0x5b, 0x70, 0xf1, 0x88, 0x1b, 0x4d, 0x23, 0xae, 0x1c, 0x37, 0x18, 0x34, 0x41, 0x6b, 0x61,
	0x73, 0xde, 0x19, 0xcf, 0x43, 0x98, 0x13, 0xdd, 0x02, 0x47, 0x0f, 0x21, 0xf6, 0x96, 0xd3, 0xc8,
	0x5b, 0xa7, 0x25, 0x55, 0xda, 0x48, 0x96, 0x88, 0x23, 0x96, 0x67, 0xe0, 0xff, 0x0a, 0x4f, 0xbd,
	0xcf, 0x12, 0xcb, 0xc3, 0x15, 0x6f, 0x79, 0xb7, 0x50, 0xed, 0x56, 0x45, 0xe8, 0x3a, 0x84, 0x63,
	0x73, 0x4f, 0x1c, 0xe2, 0xff, 0x9b, 0xa0, 0x55, 0xdb, 0xac, 0xdd, 0xbe, 0x13, 0x36, 0x46, 0xe8,
	0x23, 0x71, 0x58, 0x91, 0x58, 0xbf, 0x8f, 0x17, 0x66, 0x25, 0xcf, 0xfc, 0x3e, 0x6a, 0xc3, 0x4b,
	0xfd, 0x44, 0xa4, 0x34, 0xef, 0x46, 0x44, 0x2c, 0x49, 0xde, 0xe2, 0x5a, 0xf5, 0xf4, 0xa5, 0x9c,
	0x7d, 0x31, 0x25, 0xd1, 0x16, 0x84, 0xe5, 0x58, 0x30, 0x6c, 0x82, 0xd6, 0xd2, 0xc6, 0x6a, 0x7b,
	0x3a, 0xb7, 0xf6, 0x93, 0xd4, 0xef, 0x8d, 0x46, 0xb6, 0xa3, 0x7b, 0x7c, 0x7b, 0xfe, 0xf8, 0xeb,
	0xb5, 0x20, 0x6c, 0xc4, 0x13, 0x14, 0xad, 0xc3, 0x65, 0xc9, 0x86, 0x54, 0x79, 0x49, 0xa3, 0x03,
	0xa6, 0x14, 0x4f, 0x2c, 0x9e, 0x6b, 0x82, 0x56, 0x7d, 0x13, 0xdc, 0x0d, 0x97, 0x24, 0x1b, 0xee,
	0x7a, 0xd9, 0x1d, 0x13, 0xa8, 0x03, 0x97, 0x8d, 0x1e, 0x50, 0xc9, 0x5e, 0x69, 0x43, 0x25, 0x73,
	0x46, 0x0c, 0xf1, 0xfc, 0x2f, 0xd5, 0x19, 0x3d, 0xd8, 0xc9, 0xd9, 0x9d, 0x82, 0x44, 0xf7, 0xe1,
	0x95, 0x7c, 0xa8, 0x6f, 0x3c, 0x53, 0x4e, 0x1c, 0xf1, 0x1e, 0x1d, 0x2d, 0x89, 0xc5, 0xf5, 0xaa,
	0xeb, 0xb2, 0xb7, 0xfc, 0xe9, 0x44, 0x32, 0xba, 0x4f, 0x8b, 0x06, 0x70, 0x55, 0x7b, 0x97, 0x7a,
	0x37, 0xf6, 0xd0, 0x7e, 0xa2, 0x99, 0xa3, 0x86, 0xa9, 0x98, 0xe3, 0x46, 0x13, 0xb4, 0x16, 0x37,
	0x1e, 0x54, 0xfa, 0xfc, 0xdb, 0x1a, 0x8c, 0x05, 0x8f, 0xf3, 0x90, 0x30, 0xcf, 0x08, 0x57, 0x46,
	0xf1, 0xb3, 0xf8, 0xda, 0x36, 0x5c, 0x9e, 0xc5, 0xd0, 0x0a, 0x9c, 0x93, 0x42, 0x15, 0xbb, 0x53,
	0x1b, 0xcf, 0x30, 0x07, 0x0a, 0x9c, 0x0d, 0x8b, 0x1b, 0x2a, 0x71, 0x36, 0xdc, 0xa0, 0x70, 0x8e,
	0x0f, 0x1d, 0xba, 0x5a, 0x29, 0xf0, 0xb7, 0x8a, 0xf0, 0x97, 0x0f, 0xef, 0x3f, 0x81, 0xa2, 0x91,
	0xf5, 0x7f, 0x68, 0x24, 0xcc, 0x93, 0xb7, 0xd5, 0xc9, 0x19, 0x09, 0x4e, 0xcf, 0x48, 0x70, 0x71,
	0x46, 0xc0, 0xbb, 0x8c, 0x80, 0xcf, 0x19, 0x01, 0xc7, 0x19, 0x01, 0x27, 0x19, 0x01, 0xdf, 0x32,
	0x02, 0xbe, 0x67, 0x24, 0xb8, 0xc8, 0x08, 0xf8, 0x78, 0x4e, 0x82, 0x93, 0x73, 0x12, 0x9c, 0x9e,
	0x93, 0xe0, 0xe5, 0xbd, 0x58, 0xb8, 0x03, 0xbf, 0xdf, 0x8e, 0xb4, 0xec, 0xc4, 0x5a, 0xc7, 0x09,
	0xef, 0x94, 0x2f, 0xec, 0x4f, 0x8f, 0xfb, 0x67, 0x00, 0x00, 0x00, 0xff, 0xff, 0x71, 0x00, 0xbe,
	0x94, 0xfb, 0x03, 0x00, 0x00,
}

func (this *TensorConverterCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*TensorConverterCalculatorOptions)
	if !ok {
		that2, ok := that.(TensorConverterCalculatorOptions)
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
	if this.ZeroCenter != nil && that1.ZeroCenter != nil {
		if *this.ZeroCenter != *that1.ZeroCenter {
			return false
		}
	} else if this.ZeroCenter != nil {
		return false
	} else if that1.ZeroCenter != nil {
		return false
	}
	if this.UseCustomNormalization != nil && that1.UseCustomNormalization != nil {
		if *this.UseCustomNormalization != *that1.UseCustomNormalization {
			return false
		}
	} else if this.UseCustomNormalization != nil {
		return false
	} else if that1.UseCustomNormalization != nil {
		return false
	}
	if this.CustomDiv != nil && that1.CustomDiv != nil {
		if *this.CustomDiv != *that1.CustomDiv {
			return false
		}
	} else if this.CustomDiv != nil {
		return false
	} else if that1.CustomDiv != nil {
		return false
	}
	if this.CustomSub != nil && that1.CustomSub != nil {
		if *this.CustomSub != *that1.CustomSub {
			return false
		}
	} else if this.CustomSub != nil {
		return false
	} else if that1.CustomSub != nil {
		return false
	}
	if this.FlipVertically != nil && that1.FlipVertically != nil {
		if *this.FlipVertically != *that1.FlipVertically {
			return false
		}
	} else if this.FlipVertically != nil {
		return false
	} else if that1.FlipVertically != nil {
		return false
	}
	if this.GpuOrigin != that1.GpuOrigin {
		return false
	}
	if this.MaxNumChannels != nil && that1.MaxNumChannels != nil {
		if *this.MaxNumChannels != *that1.MaxNumChannels {
			return false
		}
	} else if this.MaxNumChannels != nil {
		return false
	} else if that1.MaxNumChannels != nil {
		return false
	}
	if this.RowMajorMatrix != nil && that1.RowMajorMatrix != nil {
		if *this.RowMajorMatrix != *that1.RowMajorMatrix {
			return false
		}
	} else if this.RowMajorMatrix != nil {
		return false
	} else if that1.RowMajorMatrix != nil {
		return false
	}
	if this.UseQuantizedTensors != nil && that1.UseQuantizedTensors != nil {
		if *this.UseQuantizedTensors != *that1.UseQuantizedTensors {
			return false
		}
	} else if this.UseQuantizedTensors != nil {
		return false
	} else if that1.UseQuantizedTensors != nil {
		return false
	}
	if !this.OutputTensorFloatRange.Equal(that1.OutputTensorFloatRange) {
		return false
	}
	return true
}
func (this *TensorConverterCalculatorOptions_TensorFloatRange) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*TensorConverterCalculatorOptions_TensorFloatRange)
	if !ok {
		that2, ok := that.(TensorConverterCalculatorOptions_TensorFloatRange)
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
	if this.Min != that1.Min {
		return false
	}
	if this.Max != that1.Max {
		return false
	}
	return true
}
func (this *TensorConverterCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 14)
	s = append(s, "&tensor.TensorConverterCalculatorOptions{")
	if this.ZeroCenter != nil {
		s = append(s, "ZeroCenter: "+valueToGoStringTensorConverterCalculator(this.ZeroCenter, "bool")+",\n")
	}
	if this.UseCustomNormalization != nil {
		s = append(s, "UseCustomNormalization: "+valueToGoStringTensorConverterCalculator(this.UseCustomNormalization, "bool")+",\n")
	}
	if this.CustomDiv != nil {
		s = append(s, "CustomDiv: "+valueToGoStringTensorConverterCalculator(this.CustomDiv, "float32")+",\n")
	}
	if this.CustomSub != nil {
		s = append(s, "CustomSub: "+valueToGoStringTensorConverterCalculator(this.CustomSub, "float32")+",\n")
	}
	if this.FlipVertically != nil {
		s = append(s, "FlipVertically: "+valueToGoStringTensorConverterCalculator(this.FlipVertically, "bool")+",\n")
	}
	s = append(s, "GpuOrigin: "+fmt.Sprintf("%#v", this.GpuOrigin)+",\n")
	if this.MaxNumChannels != nil {
		s = append(s, "MaxNumChannels: "+valueToGoStringTensorConverterCalculator(this.MaxNumChannels, "int32")+",\n")
	}
	if this.RowMajorMatrix != nil {
		s = append(s, "RowMajorMatrix: "+valueToGoStringTensorConverterCalculator(this.RowMajorMatrix, "bool")+",\n")
	}
	if this.UseQuantizedTensors != nil {
		s = append(s, "UseQuantizedTensors: "+valueToGoStringTensorConverterCalculator(this.UseQuantizedTensors, "bool")+",\n")
	}
	if this.OutputTensorFloatRange != nil {
		s = append(s, "OutputTensorFloatRange: "+fmt.Sprintf("%#v", this.OutputTensorFloatRange)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *TensorConverterCalculatorOptions_TensorFloatRange) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&tensor.TensorConverterCalculatorOptions_TensorFloatRange{")
	s = append(s, "Min: "+fmt.Sprintf("%#v", this.Min)+",\n")
	s = append(s, "Max: "+fmt.Sprintf("%#v", this.Max)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringTensorConverterCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *TensorConverterCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TensorConverterCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *TensorConverterCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i = encodeVarintTensorConverterCalculator(dAtA, i, uint64(m.GpuOrigin))
	i--
	dAtA[i] = 0x50
	if m.OutputTensorFloatRange != nil {
		{
			size, err := m.OutputTensorFloatRange.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintTensorConverterCalculator(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x4a
	}
	if m.CustomSub != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.CustomSub))))
		i--
		dAtA[i] = 0x45
	}
	if m.CustomDiv != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.CustomDiv))))
		i--
		dAtA[i] = 0x3d
	}
	if m.UseCustomNormalization != nil {
		i--
		if *m.UseCustomNormalization {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x30
	}
	if m.UseQuantizedTensors != nil {
		i--
		if *m.UseQuantizedTensors {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x28
	}
	if m.RowMajorMatrix != nil {
		i--
		if *m.RowMajorMatrix {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x20
	}
	if m.MaxNumChannels != nil {
		i = encodeVarintTensorConverterCalculator(dAtA, i, uint64(*m.MaxNumChannels))
		i--
		dAtA[i] = 0x18
	}
	if m.FlipVertically != nil {
		i--
		if *m.FlipVertically {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x10
	}
	if m.ZeroCenter != nil {
		i--
		if *m.ZeroCenter {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func (m *TensorConverterCalculatorOptions_TensorFloatRange) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TensorConverterCalculatorOptions_TensorFloatRange) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *TensorConverterCalculatorOptions_TensorFloatRange) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Max))))
	i--
	dAtA[i] = 0x15
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Min))))
	i--
	dAtA[i] = 0xd
	return len(dAtA) - i, nil
}

func encodeVarintTensorConverterCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovTensorConverterCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *TensorConverterCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.ZeroCenter != nil {
		n += 2
	}
	if m.FlipVertically != nil {
		n += 2
	}
	if m.MaxNumChannels != nil {
		n += 1 + sovTensorConverterCalculator(uint64(*m.MaxNumChannels))
	}
	if m.RowMajorMatrix != nil {
		n += 2
	}
	if m.UseQuantizedTensors != nil {
		n += 2
	}
	if m.UseCustomNormalization != nil {
		n += 2
	}
	if m.CustomDiv != nil {
		n += 5
	}
	if m.CustomSub != nil {
		n += 5
	}
	if m.OutputTensorFloatRange != nil {
		l = m.OutputTensorFloatRange.Size()
		n += 1 + l + sovTensorConverterCalculator(uint64(l))
	}
	n += 1 + sovTensorConverterCalculator(uint64(m.GpuOrigin))
	return n
}

func (m *TensorConverterCalculatorOptions_TensorFloatRange) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 5
	n += 5
	return n
}

func sovTensorConverterCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozTensorConverterCalculator(x uint64) (n int) {
	return sovTensorConverterCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *TensorConverterCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&TensorConverterCalculatorOptions{`,
		`ZeroCenter:` + valueToStringTensorConverterCalculator(this.ZeroCenter) + `,`,
		`FlipVertically:` + valueToStringTensorConverterCalculator(this.FlipVertically) + `,`,
		`MaxNumChannels:` + valueToStringTensorConverterCalculator(this.MaxNumChannels) + `,`,
		`RowMajorMatrix:` + valueToStringTensorConverterCalculator(this.RowMajorMatrix) + `,`,
		`UseQuantizedTensors:` + valueToStringTensorConverterCalculator(this.UseQuantizedTensors) + `,`,
		`UseCustomNormalization:` + valueToStringTensorConverterCalculator(this.UseCustomNormalization) + `,`,
		`CustomDiv:` + valueToStringTensorConverterCalculator(this.CustomDiv) + `,`,
		`CustomSub:` + valueToStringTensorConverterCalculator(this.CustomSub) + `,`,
		`OutputTensorFloatRange:` + strings.Replace(fmt.Sprintf("%v", this.OutputTensorFloatRange), "TensorConverterCalculatorOptions_TensorFloatRange", "TensorConverterCalculatorOptions_TensorFloatRange", 1) + `,`,
		`GpuOrigin:` + fmt.Sprintf("%v", this.GpuOrigin) + `,`,
		`}`,
	}, "")
	return s
}
func (this *TensorConverterCalculatorOptions_TensorFloatRange) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&TensorConverterCalculatorOptions_TensorFloatRange{`,
		`Min:` + fmt.Sprintf("%v", this.Min) + `,`,
		`Max:` + fmt.Sprintf("%v", this.Max) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringTensorConverterCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *TensorConverterCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorConverterCalculator
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
			return fmt.Errorf("proto: TensorConverterCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TensorConverterCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field ZeroCenter", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
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
			m.ZeroCenter = &b
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field FlipVertically", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
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
			m.FlipVertically = &b
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field MaxNumChannels", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.MaxNumChannels = &v
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RowMajorMatrix", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
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
			m.RowMajorMatrix = &b
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field UseQuantizedTensors", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
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
			m.UseQuantizedTensors = &b
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field UseCustomNormalization", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
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
			m.UseCustomNormalization = &b
		case 7:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field CustomDiv", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.CustomDiv = &v2
		case 8:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field CustomSub", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.CustomSub = &v2
		case 9:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutputTensorFloatRange", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
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
				return ErrInvalidLengthTensorConverterCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthTensorConverterCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.OutputTensorFloatRange == nil {
				m.OutputTensorFloatRange = &TensorConverterCalculatorOptions_TensorFloatRange{}
			}
			if err := m.OutputTensorFloatRange.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 10:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field GpuOrigin", wireType)
			}
			m.GpuOrigin = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorConverterCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.GpuOrigin |= gpu.GpuOrigin_Mode(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipTensorConverterCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTensorConverterCalculator
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
func (m *TensorConverterCalculatorOptions_TensorFloatRange) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorConverterCalculator
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
			return fmt.Errorf("proto: TensorFloatRange: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TensorFloatRange: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Min", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Min = float32(math.Float32frombits(v))
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Max", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Max = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipTensorConverterCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTensorConverterCalculator
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
func skipTensorConverterCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTensorConverterCalculator
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
					return 0, ErrIntOverflowTensorConverterCalculator
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
					return 0, ErrIntOverflowTensorConverterCalculator
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
				return 0, ErrInvalidLengthTensorConverterCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupTensorConverterCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthTensorConverterCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthTensorConverterCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTensorConverterCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupTensorConverterCalculator = fmt.Errorf("proto: unexpected end of group")
)