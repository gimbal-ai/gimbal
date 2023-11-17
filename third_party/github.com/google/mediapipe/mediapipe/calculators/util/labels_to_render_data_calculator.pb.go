// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/util/labels_to_render_data_calculator.proto

package util

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	framework "github.com/google/mediapipe/mediapipe/framework"
	util "github.com/google/mediapipe/mediapipe/util"
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

type LabelsToRenderDataCalculatorOptions_Location int32

const (
	TOP_LEFT    LabelsToRenderDataCalculatorOptions_Location = 0
	BOTTOM_LEFT LabelsToRenderDataCalculatorOptions_Location = 1
)

var LabelsToRenderDataCalculatorOptions_Location_name = map[int32]string{
	0: "TOP_LEFT",
	1: "BOTTOM_LEFT",
}

var LabelsToRenderDataCalculatorOptions_Location_value = map[string]int32{
	"TOP_LEFT":    0,
	"BOTTOM_LEFT": 1,
}

func (x LabelsToRenderDataCalculatorOptions_Location) Enum() *LabelsToRenderDataCalculatorOptions_Location {
	p := new(LabelsToRenderDataCalculatorOptions_Location)
	*p = x
	return p
}

func (x LabelsToRenderDataCalculatorOptions_Location) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(LabelsToRenderDataCalculatorOptions_Location_name, int32(x))
}

func (x *LabelsToRenderDataCalculatorOptions_Location) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(LabelsToRenderDataCalculatorOptions_Location_value, data, "LabelsToRenderDataCalculatorOptions_Location")
	if err != nil {
		return err
	}
	*x = LabelsToRenderDataCalculatorOptions_Location(value)
	return nil
}

func (LabelsToRenderDataCalculatorOptions_Location) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_fab7e4be0afd3606, []int{0, 0}
}

type LabelsToRenderDataCalculatorOptions struct {
	Color                      []*util.Color                                 `protobuf:"bytes,1,rep,name=color" json:"color,omitempty"`
	Thickness                  *float64                                      `protobuf:"fixed64,2,opt,name=thickness,def=2" json:"thickness,omitempty"`
	OutlineColor               []*util.Color                                 `protobuf:"bytes,12,rep,name=outline_color,json=outlineColor" json:"outline_color,omitempty"`
	OutlineThickness           float64                                       `protobuf:"fixed64,11,opt,name=outline_thickness,json=outlineThickness" json:"outline_thickness"`
	FontHeightPx               *int32                                        `protobuf:"varint,3,opt,name=font_height_px,json=fontHeightPx,def=50" json:"font_height_px,omitempty"`
	HorizontalOffsetPx         *int32                                        `protobuf:"varint,7,opt,name=horizontal_offset_px,json=horizontalOffsetPx,def=0" json:"horizontal_offset_px,omitempty"`
	VerticalOffsetPx           *int32                                        `protobuf:"varint,8,opt,name=vertical_offset_px,json=verticalOffsetPx,def=0" json:"vertical_offset_px,omitempty"`
	MaxNumLabels               *int32                                        `protobuf:"varint,4,opt,name=max_num_labels,json=maxNumLabels,def=1" json:"max_num_labels,omitempty"`
	FontFace                   *int32                                        `protobuf:"varint,5,opt,name=font_face,json=fontFace,def=0" json:"font_face,omitempty"`
	Location                   *LabelsToRenderDataCalculatorOptions_Location `protobuf:"varint,6,opt,name=location,enum=mediapipe.LabelsToRenderDataCalculatorOptions_Location,def=0" json:"location,omitempty"`
	UseDisplayName             *bool                                         `protobuf:"varint,9,opt,name=use_display_name,json=useDisplayName,def=0" json:"use_display_name,omitempty"`
	DisplayClassificationScore *bool                                         `protobuf:"varint,10,opt,name=display_classification_score,json=displayClassificationScore,def=0" json:"display_classification_score,omitempty"`
}

func (m *LabelsToRenderDataCalculatorOptions) Reset()      { *m = LabelsToRenderDataCalculatorOptions{} }
func (*LabelsToRenderDataCalculatorOptions) ProtoMessage() {}
func (*LabelsToRenderDataCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_fab7e4be0afd3606, []int{0}
}
func (m *LabelsToRenderDataCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *LabelsToRenderDataCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_LabelsToRenderDataCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *LabelsToRenderDataCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LabelsToRenderDataCalculatorOptions.Merge(m, src)
}
func (m *LabelsToRenderDataCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *LabelsToRenderDataCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_LabelsToRenderDataCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_LabelsToRenderDataCalculatorOptions proto.InternalMessageInfo

const Default_LabelsToRenderDataCalculatorOptions_Thickness float64 = 2
const Default_LabelsToRenderDataCalculatorOptions_FontHeightPx int32 = 50
const Default_LabelsToRenderDataCalculatorOptions_HorizontalOffsetPx int32 = 0
const Default_LabelsToRenderDataCalculatorOptions_VerticalOffsetPx int32 = 0
const Default_LabelsToRenderDataCalculatorOptions_MaxNumLabels int32 = 1
const Default_LabelsToRenderDataCalculatorOptions_FontFace int32 = 0
const Default_LabelsToRenderDataCalculatorOptions_Location LabelsToRenderDataCalculatorOptions_Location = TOP_LEFT
const Default_LabelsToRenderDataCalculatorOptions_UseDisplayName bool = false
const Default_LabelsToRenderDataCalculatorOptions_DisplayClassificationScore bool = false

func (m *LabelsToRenderDataCalculatorOptions) GetColor() []*util.Color {
	if m != nil {
		return m.Color
	}
	return nil
}

func (m *LabelsToRenderDataCalculatorOptions) GetThickness() float64 {
	if m != nil && m.Thickness != nil {
		return *m.Thickness
	}
	return Default_LabelsToRenderDataCalculatorOptions_Thickness
}

func (m *LabelsToRenderDataCalculatorOptions) GetOutlineColor() []*util.Color {
	if m != nil {
		return m.OutlineColor
	}
	return nil
}

func (m *LabelsToRenderDataCalculatorOptions) GetOutlineThickness() float64 {
	if m != nil {
		return m.OutlineThickness
	}
	return 0
}

func (m *LabelsToRenderDataCalculatorOptions) GetFontHeightPx() int32 {
	if m != nil && m.FontHeightPx != nil {
		return *m.FontHeightPx
	}
	return Default_LabelsToRenderDataCalculatorOptions_FontHeightPx
}

func (m *LabelsToRenderDataCalculatorOptions) GetHorizontalOffsetPx() int32 {
	if m != nil && m.HorizontalOffsetPx != nil {
		return *m.HorizontalOffsetPx
	}
	return Default_LabelsToRenderDataCalculatorOptions_HorizontalOffsetPx
}

func (m *LabelsToRenderDataCalculatorOptions) GetVerticalOffsetPx() int32 {
	if m != nil && m.VerticalOffsetPx != nil {
		return *m.VerticalOffsetPx
	}
	return Default_LabelsToRenderDataCalculatorOptions_VerticalOffsetPx
}

func (m *LabelsToRenderDataCalculatorOptions) GetMaxNumLabels() int32 {
	if m != nil && m.MaxNumLabels != nil {
		return *m.MaxNumLabels
	}
	return Default_LabelsToRenderDataCalculatorOptions_MaxNumLabels
}

func (m *LabelsToRenderDataCalculatorOptions) GetFontFace() int32 {
	if m != nil && m.FontFace != nil {
		return *m.FontFace
	}
	return Default_LabelsToRenderDataCalculatorOptions_FontFace
}

func (m *LabelsToRenderDataCalculatorOptions) GetLocation() LabelsToRenderDataCalculatorOptions_Location {
	if m != nil && m.Location != nil {
		return *m.Location
	}
	return Default_LabelsToRenderDataCalculatorOptions_Location
}

func (m *LabelsToRenderDataCalculatorOptions) GetUseDisplayName() bool {
	if m != nil && m.UseDisplayName != nil {
		return *m.UseDisplayName
	}
	return Default_LabelsToRenderDataCalculatorOptions_UseDisplayName
}

func (m *LabelsToRenderDataCalculatorOptions) GetDisplayClassificationScore() bool {
	if m != nil && m.DisplayClassificationScore != nil {
		return *m.DisplayClassificationScore
	}
	return Default_LabelsToRenderDataCalculatorOptions_DisplayClassificationScore
}

var E_LabelsToRenderDataCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*LabelsToRenderDataCalculatorOptions)(nil),
	Field:         271660364,
	Name:          "mediapipe.LabelsToRenderDataCalculatorOptions.ext",
	Tag:           "bytes,271660364,opt,name=ext",
	Filename:      "mediapipe/calculators/util/labels_to_render_data_calculator.proto",
}

func init() {
	proto.RegisterEnum("mediapipe.LabelsToRenderDataCalculatorOptions_Location", LabelsToRenderDataCalculatorOptions_Location_name, LabelsToRenderDataCalculatorOptions_Location_value)
	proto.RegisterExtension(E_LabelsToRenderDataCalculatorOptions_Ext)
	proto.RegisterType((*LabelsToRenderDataCalculatorOptions)(nil), "mediapipe.LabelsToRenderDataCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/util/labels_to_render_data_calculator.proto", fileDescriptor_fab7e4be0afd3606)
}

var fileDescriptor_fab7e4be0afd3606 = []byte{
	// 600 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x93, 0x31, 0x6f, 0xd3, 0x4e,
	0x18, 0xc6, 0x7d, 0x6d, 0xd3, 0x7f, 0x72, 0xc9, 0x3f, 0x84, 0x13, 0x83, 0x15, 0x55, 0xd7, 0xa8,
	0x20, 0x30, 0x4b, 0xdc, 0x06, 0x15, 0xa4, 0x6c, 0xb4, 0xa5, 0x30, 0x94, 0xa6, 0x32, 0x99, 0x90,
	0xd0, 0xe9, 0xea, 0x9c, 0x93, 0x53, 0xcf, 0xbe, 0xc8, 0x77, 0x86, 0xc0, 0x04, 0xdf, 0x80, 0x95,
	0x6f, 0xc0, 0x07, 0x61, 0xe8, 0xc0, 0xd0, 0xb1, 0x13, 0xa2, 0xee, 0x82, 0x98, 0xfa, 0x11, 0x90,
	0xed, 0xd8, 0x6e, 0x2b, 0x55, 0x82, 0xd1, 0xef, 0xf3, 0x3c, 0xbf, 0xf7, 0xf5, 0xbd, 0x77, 0xf0,
	0xa9, 0xcf, 0x46, 0x9c, 0x4e, 0xf9, 0x94, 0xd9, 0x2e, 0x15, 0x6e, 0x24, 0xa8, 0x96, 0xa1, 0xb2,
	0x23, 0xcd, 0x85, 0x2d, 0xe8, 0x21, 0x13, 0x8a, 0x68, 0x49, 0x42, 0x16, 0x8c, 0x58, 0x48, 0x46,
	0x54, 0x53, 0x52, 0xda, 0xba, 0xd3, 0x50, 0x6a, 0x89, 0x6a, 0x05, 0xa2, 0x7d, 0xaf, 0xa4, 0x79,
	0x21, 0xf5, 0xd9, 0x3b, 0x19, 0x1e, 0xd9, 0xd7, 0x03, 0xed, 0x76, 0xe9, 0x4a, 0xfb, 0xb8, 0x52,
	0xe4, 0xda, 0xda, 0x97, 0x65, 0x78, 0x77, 0x2f, 0xed, 0x3b, 0x94, 0x4e, 0xda, 0x75, 0x87, 0x6a,
	0xba, 0x5d, 0x20, 0x06, 0x53, 0xcd, 0x65, 0xa0, 0xd0, 0x7d, 0x58, 0x49, 0x63, 0x26, 0xe8, 0x2c,
	0x5a, 0xf5, 0x5e, 0xab, 0x5b, 0x30, 0xbb, 0xdb, 0x49, 0xdd, 0xc9, 0x64, 0xb4, 0x0a, 0x6b, 0x7a,
	0xc2, 0xdd, 0xa3, 0x80, 0x29, 0x65, 0x2e, 0x74, 0x80, 0x05, 0xfa, 0xa0, 0xe7, 0x94, 0x35, 0xb4,
	0x09, 0xff, 0x97, 0x91, 0x16, 0x3c, 0x60, 0x24, 0x03, 0x36, 0x6e, 0x00, 0x36, 0xe6, 0xb6, 0xf4,
	0x0b, 0x6d, 0xc0, 0xdb, 0x79, 0xac, 0xe4, 0xd7, 0x13, 0xfe, 0xd6, 0xd2, 0xf1, 0x8f, 0x55, 0xc3,
	0x69, 0xcd, 0xe5, 0x61, 0xd1, 0xc9, 0x82, 0x4d, 0x4f, 0x06, 0x9a, 0x4c, 0x18, 0x1f, 0x4f, 0x34,
	0x99, 0xce, 0xcc, 0xc5, 0x0e, 0xb0, 0x2a, 0xfd, 0x85, 0xcd, 0x75, 0xa7, 0x91, 0x28, 0x2f, 0x52,
	0xe1, 0x60, 0x86, 0x1e, 0xc1, 0x3b, 0x13, 0x19, 0xf2, 0x0f, 0x32, 0xd0, 0x54, 0x10, 0xe9, 0x79,
	0x8a, 0xa5, 0xfe, 0xff, 0x52, 0x3f, 0x58, 0x77, 0x50, 0x29, 0x0f, 0x52, 0xf5, 0x60, 0x86, 0x6c,
	0x88, 0xde, 0xb2, 0x50, 0x73, 0xf7, 0x4a, 0xa4, 0x9a, 0x47, 0x5a, 0xb9, 0x58, 0x04, 0x1e, 0xc0,
	0xa6, 0x4f, 0x67, 0x24, 0x88, 0x7c, 0x92, 0x6d, 0xda, 0x5c, 0xca, 0xcc, 0x1b, 0x4e, 0xc3, 0xa7,
	0xb3, 0xfd, 0xc8, 0xcf, 0x16, 0x81, 0x30, 0xac, 0xa5, 0x83, 0x7b, 0xd4, 0x65, 0x66, 0x25, 0x07,
	0x56, 0x93, 0xda, 0x2e, 0x75, 0x19, 0x7a, 0x03, 0xab, 0x42, 0xba, 0x34, 0x59, 0x8c, 0xb9, 0xdc,
	0x01, 0x56, 0xb3, 0xf7, 0xe4, 0xd2, 0xe9, 0xfd, 0xc5, 0x36, 0xbb, 0x7b, 0xf3, 0x78, 0xbf, 0x3a,
	0x1c, 0x1c, 0x90, 0xbd, 0x67, 0xbb, 0x43, 0xa7, 0x40, 0x22, 0x1b, 0xb6, 0x22, 0xc5, 0xc8, 0x88,
	0xab, 0xa9, 0xa0, 0xef, 0x49, 0x40, 0x7d, 0x66, 0xd6, 0x3a, 0xc0, 0xaa, 0xf6, 0x2b, 0x1e, 0x15,
	0x8a, 0x39, 0xcd, 0x48, 0xb1, 0x9d, 0x4c, 0xdd, 0xa7, 0x3e, 0x43, 0xcf, 0xe1, 0x4a, 0x6e, 0x76,
	0x05, 0x55, 0x8a, 0x7b, 0x3c, 0x43, 0x11, 0xe5, 0xca, 0x90, 0x99, 0xf0, 0x72, 0xb8, 0x3d, 0xb7,
	0x6e, 0x5f, 0x71, 0xbe, 0x4a, 0x8c, 0x6b, 0x0f, 0x61, 0x35, 0x9f, 0x0c, 0x35, 0x60, 0x31, 0x5b,
	0xcb, 0x40, 0xb7, 0x60, 0x7d, 0x6b, 0x30, 0x1c, 0x0e, 0x5e, 0x66, 0x05, 0xd0, 0x3b, 0x84, 0x8b,
	0x6c, 0xa6, 0xd1, 0xca, 0xe5, 0x6b, 0x73, 0xfd, 0x37, 0xcd, 0xef, 0xbf, 0xbf, 0x7d, 0x02, 0x1d,
	0x60, 0xd5, 0x7b, 0xdd, 0x7f, 0x3b, 0x20, 0x27, 0x81, 0x6f, 0x89, 0x93, 0x33, 0x6c, 0x9c, 0x9e,
	0x61, 0xe3, 0xe2, 0x0c, 0x83, 0x8f, 0x31, 0x06, 0x5f, 0x63, 0x0c, 0x8e, 0x63, 0x0c, 0x4e, 0x62,
	0x0c, 0x7e, 0xc6, 0x18, 0xfc, 0x8a, 0xb1, 0x71, 0x11, 0x63, 0xf0, 0xf9, 0x1c, 0x1b, 0x27, 0xe7,
	0xd8, 0x38, 0x3d, 0xc7, 0xc6, 0xeb, 0xc7, 0x63, 0xae, 0x27, 0xd1, 0x61, 0xd7, 0x95, 0xbe, 0x3d,
	0x96, 0x72, 0x2c, 0x98, 0x5d, 0xbe, 0xbf, 0x9b, 0x5f, 0xff, 0x9f, 0x00, 0x00, 0x00, 0xff, 0xff,
	0xc3, 0xd3, 0xf2, 0x95, 0x1a, 0x04, 0x00, 0x00,
}

func (x LabelsToRenderDataCalculatorOptions_Location) String() string {
	s, ok := LabelsToRenderDataCalculatorOptions_Location_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *LabelsToRenderDataCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*LabelsToRenderDataCalculatorOptions)
	if !ok {
		that2, ok := that.(LabelsToRenderDataCalculatorOptions)
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
	if len(this.Color) != len(that1.Color) {
		return false
	}
	for i := range this.Color {
		if !this.Color[i].Equal(that1.Color[i]) {
			return false
		}
	}
	if this.Thickness != nil && that1.Thickness != nil {
		if *this.Thickness != *that1.Thickness {
			return false
		}
	} else if this.Thickness != nil {
		return false
	} else if that1.Thickness != nil {
		return false
	}
	if len(this.OutlineColor) != len(that1.OutlineColor) {
		return false
	}
	for i := range this.OutlineColor {
		if !this.OutlineColor[i].Equal(that1.OutlineColor[i]) {
			return false
		}
	}
	if this.OutlineThickness != that1.OutlineThickness {
		return false
	}
	if this.FontHeightPx != nil && that1.FontHeightPx != nil {
		if *this.FontHeightPx != *that1.FontHeightPx {
			return false
		}
	} else if this.FontHeightPx != nil {
		return false
	} else if that1.FontHeightPx != nil {
		return false
	}
	if this.HorizontalOffsetPx != nil && that1.HorizontalOffsetPx != nil {
		if *this.HorizontalOffsetPx != *that1.HorizontalOffsetPx {
			return false
		}
	} else if this.HorizontalOffsetPx != nil {
		return false
	} else if that1.HorizontalOffsetPx != nil {
		return false
	}
	if this.VerticalOffsetPx != nil && that1.VerticalOffsetPx != nil {
		if *this.VerticalOffsetPx != *that1.VerticalOffsetPx {
			return false
		}
	} else if this.VerticalOffsetPx != nil {
		return false
	} else if that1.VerticalOffsetPx != nil {
		return false
	}
	if this.MaxNumLabels != nil && that1.MaxNumLabels != nil {
		if *this.MaxNumLabels != *that1.MaxNumLabels {
			return false
		}
	} else if this.MaxNumLabels != nil {
		return false
	} else if that1.MaxNumLabels != nil {
		return false
	}
	if this.FontFace != nil && that1.FontFace != nil {
		if *this.FontFace != *that1.FontFace {
			return false
		}
	} else if this.FontFace != nil {
		return false
	} else if that1.FontFace != nil {
		return false
	}
	if this.Location != nil && that1.Location != nil {
		if *this.Location != *that1.Location {
			return false
		}
	} else if this.Location != nil {
		return false
	} else if that1.Location != nil {
		return false
	}
	if this.UseDisplayName != nil && that1.UseDisplayName != nil {
		if *this.UseDisplayName != *that1.UseDisplayName {
			return false
		}
	} else if this.UseDisplayName != nil {
		return false
	} else if that1.UseDisplayName != nil {
		return false
	}
	if this.DisplayClassificationScore != nil && that1.DisplayClassificationScore != nil {
		if *this.DisplayClassificationScore != *that1.DisplayClassificationScore {
			return false
		}
	} else if this.DisplayClassificationScore != nil {
		return false
	} else if that1.DisplayClassificationScore != nil {
		return false
	}
	return true
}
func (this *LabelsToRenderDataCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 16)
	s = append(s, "&util.LabelsToRenderDataCalculatorOptions{")
	if this.Color != nil {
		s = append(s, "Color: "+fmt.Sprintf("%#v", this.Color)+",\n")
	}
	if this.Thickness != nil {
		s = append(s, "Thickness: "+valueToGoStringLabelsToRenderDataCalculator(this.Thickness, "float64")+",\n")
	}
	if this.OutlineColor != nil {
		s = append(s, "OutlineColor: "+fmt.Sprintf("%#v", this.OutlineColor)+",\n")
	}
	s = append(s, "OutlineThickness: "+fmt.Sprintf("%#v", this.OutlineThickness)+",\n")
	if this.FontHeightPx != nil {
		s = append(s, "FontHeightPx: "+valueToGoStringLabelsToRenderDataCalculator(this.FontHeightPx, "int32")+",\n")
	}
	if this.HorizontalOffsetPx != nil {
		s = append(s, "HorizontalOffsetPx: "+valueToGoStringLabelsToRenderDataCalculator(this.HorizontalOffsetPx, "int32")+",\n")
	}
	if this.VerticalOffsetPx != nil {
		s = append(s, "VerticalOffsetPx: "+valueToGoStringLabelsToRenderDataCalculator(this.VerticalOffsetPx, "int32")+",\n")
	}
	if this.MaxNumLabels != nil {
		s = append(s, "MaxNumLabels: "+valueToGoStringLabelsToRenderDataCalculator(this.MaxNumLabels, "int32")+",\n")
	}
	if this.FontFace != nil {
		s = append(s, "FontFace: "+valueToGoStringLabelsToRenderDataCalculator(this.FontFace, "int32")+",\n")
	}
	if this.Location != nil {
		s = append(s, "Location: "+valueToGoStringLabelsToRenderDataCalculator(this.Location, "LabelsToRenderDataCalculatorOptions_Location")+",\n")
	}
	if this.UseDisplayName != nil {
		s = append(s, "UseDisplayName: "+valueToGoStringLabelsToRenderDataCalculator(this.UseDisplayName, "bool")+",\n")
	}
	if this.DisplayClassificationScore != nil {
		s = append(s, "DisplayClassificationScore: "+valueToGoStringLabelsToRenderDataCalculator(this.DisplayClassificationScore, "bool")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringLabelsToRenderDataCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *LabelsToRenderDataCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *LabelsToRenderDataCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *LabelsToRenderDataCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.OutlineColor) > 0 {
		for iNdEx := len(m.OutlineColor) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.OutlineColor[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0x62
		}
	}
	i -= 8
	encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(m.OutlineThickness))))
	i--
	dAtA[i] = 0x59
	if m.DisplayClassificationScore != nil {
		i--
		if *m.DisplayClassificationScore {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x50
	}
	if m.UseDisplayName != nil {
		i--
		if *m.UseDisplayName {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x48
	}
	if m.VerticalOffsetPx != nil {
		i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(*m.VerticalOffsetPx))
		i--
		dAtA[i] = 0x40
	}
	if m.HorizontalOffsetPx != nil {
		i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(*m.HorizontalOffsetPx))
		i--
		dAtA[i] = 0x38
	}
	if m.Location != nil {
		i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(*m.Location))
		i--
		dAtA[i] = 0x30
	}
	if m.FontFace != nil {
		i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(*m.FontFace))
		i--
		dAtA[i] = 0x28
	}
	if m.MaxNumLabels != nil {
		i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(*m.MaxNumLabels))
		i--
		dAtA[i] = 0x20
	}
	if m.FontHeightPx != nil {
		i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(*m.FontHeightPx))
		i--
		dAtA[i] = 0x18
	}
	if m.Thickness != nil {
		i -= 8
		encoding_binary.LittleEndian.PutUint64(dAtA[i:], uint64(math.Float64bits(float64(*m.Thickness))))
		i--
		dAtA[i] = 0x11
	}
	if len(m.Color) > 0 {
		for iNdEx := len(m.Color) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.Color[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintLabelsToRenderDataCalculator(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintLabelsToRenderDataCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovLabelsToRenderDataCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *LabelsToRenderDataCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Color) > 0 {
		for _, e := range m.Color {
			l = e.Size()
			n += 1 + l + sovLabelsToRenderDataCalculator(uint64(l))
		}
	}
	if m.Thickness != nil {
		n += 9
	}
	if m.FontHeightPx != nil {
		n += 1 + sovLabelsToRenderDataCalculator(uint64(*m.FontHeightPx))
	}
	if m.MaxNumLabels != nil {
		n += 1 + sovLabelsToRenderDataCalculator(uint64(*m.MaxNumLabels))
	}
	if m.FontFace != nil {
		n += 1 + sovLabelsToRenderDataCalculator(uint64(*m.FontFace))
	}
	if m.Location != nil {
		n += 1 + sovLabelsToRenderDataCalculator(uint64(*m.Location))
	}
	if m.HorizontalOffsetPx != nil {
		n += 1 + sovLabelsToRenderDataCalculator(uint64(*m.HorizontalOffsetPx))
	}
	if m.VerticalOffsetPx != nil {
		n += 1 + sovLabelsToRenderDataCalculator(uint64(*m.VerticalOffsetPx))
	}
	if m.UseDisplayName != nil {
		n += 2
	}
	if m.DisplayClassificationScore != nil {
		n += 2
	}
	n += 9
	if len(m.OutlineColor) > 0 {
		for _, e := range m.OutlineColor {
			l = e.Size()
			n += 1 + l + sovLabelsToRenderDataCalculator(uint64(l))
		}
	}
	return n
}

func sovLabelsToRenderDataCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozLabelsToRenderDataCalculator(x uint64) (n int) {
	return sovLabelsToRenderDataCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *LabelsToRenderDataCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForColor := "[]*Color{"
	for _, f := range this.Color {
		repeatedStringForColor += strings.Replace(fmt.Sprintf("%v", f), "Color", "util.Color", 1) + ","
	}
	repeatedStringForColor += "}"
	repeatedStringForOutlineColor := "[]*Color{"
	for _, f := range this.OutlineColor {
		repeatedStringForOutlineColor += strings.Replace(fmt.Sprintf("%v", f), "Color", "util.Color", 1) + ","
	}
	repeatedStringForOutlineColor += "}"
	s := strings.Join([]string{`&LabelsToRenderDataCalculatorOptions{`,
		`Color:` + repeatedStringForColor + `,`,
		`Thickness:` + valueToStringLabelsToRenderDataCalculator(this.Thickness) + `,`,
		`FontHeightPx:` + valueToStringLabelsToRenderDataCalculator(this.FontHeightPx) + `,`,
		`MaxNumLabels:` + valueToStringLabelsToRenderDataCalculator(this.MaxNumLabels) + `,`,
		`FontFace:` + valueToStringLabelsToRenderDataCalculator(this.FontFace) + `,`,
		`Location:` + valueToStringLabelsToRenderDataCalculator(this.Location) + `,`,
		`HorizontalOffsetPx:` + valueToStringLabelsToRenderDataCalculator(this.HorizontalOffsetPx) + `,`,
		`VerticalOffsetPx:` + valueToStringLabelsToRenderDataCalculator(this.VerticalOffsetPx) + `,`,
		`UseDisplayName:` + valueToStringLabelsToRenderDataCalculator(this.UseDisplayName) + `,`,
		`DisplayClassificationScore:` + valueToStringLabelsToRenderDataCalculator(this.DisplayClassificationScore) + `,`,
		`OutlineThickness:` + fmt.Sprintf("%v", this.OutlineThickness) + `,`,
		`OutlineColor:` + repeatedStringForOutlineColor + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringLabelsToRenderDataCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *LabelsToRenderDataCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLabelsToRenderDataCalculator
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
			return fmt.Errorf("proto: LabelsToRenderDataCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: LabelsToRenderDataCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Color", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
				return ErrInvalidLengthLabelsToRenderDataCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthLabelsToRenderDataCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Color = append(m.Color, &util.Color{})
			if err := m.Color[len(m.Color)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field Thickness", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			v2 := float64(math.Float64frombits(v))
			m.Thickness = &v2
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field FontHeightPx", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
			m.FontHeightPx = &v
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field MaxNumLabels", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
			m.MaxNumLabels = &v
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field FontFace", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
			m.FontFace = &v
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Location", wireType)
			}
			var v LabelsToRenderDataCalculatorOptions_Location
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				v |= LabelsToRenderDataCalculatorOptions_Location(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			m.Location = &v
		case 7:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field HorizontalOffsetPx", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
			m.HorizontalOffsetPx = &v
		case 8:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field VerticalOffsetPx", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
			m.VerticalOffsetPx = &v
		case 9:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field UseDisplayName", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
			m.UseDisplayName = &b
		case 10:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field DisplayClassificationScore", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
			m.DisplayClassificationScore = &b
		case 11:
			if wireType != 1 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutlineThickness", wireType)
			}
			var v uint64
			if (iNdEx + 8) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint64(encoding_binary.LittleEndian.Uint64(dAtA[iNdEx:]))
			iNdEx += 8
			m.OutlineThickness = float64(math.Float64frombits(v))
		case 12:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutlineColor", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLabelsToRenderDataCalculator
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
				return ErrInvalidLengthLabelsToRenderDataCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthLabelsToRenderDataCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.OutlineColor = append(m.OutlineColor, &util.Color{})
			if err := m.OutlineColor[len(m.OutlineColor)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipLabelsToRenderDataCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthLabelsToRenderDataCalculator
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
func skipLabelsToRenderDataCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowLabelsToRenderDataCalculator
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
					return 0, ErrIntOverflowLabelsToRenderDataCalculator
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
					return 0, ErrIntOverflowLabelsToRenderDataCalculator
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
				return 0, ErrInvalidLengthLabelsToRenderDataCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupLabelsToRenderDataCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthLabelsToRenderDataCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthLabelsToRenderDataCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowLabelsToRenderDataCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupLabelsToRenderDataCalculator = fmt.Errorf("proto: unexpected end of group")
)