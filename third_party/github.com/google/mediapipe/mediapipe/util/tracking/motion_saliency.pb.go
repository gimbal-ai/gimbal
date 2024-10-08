// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/util/tracking/motion_saliency.proto

package tracking

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

type MotionSaliencyOptions struct {
	BoundLeft                  *float32 `protobuf:"fixed32,1,opt,name=bound_left,json=boundLeft,def=0.3" json:"bound_left,omitempty"`
	BoundBottom                *float32 `protobuf:"fixed32,2,opt,name=bound_bottom,json=boundBottom,def=0.3" json:"bound_bottom,omitempty"`
	BoundRight                 *float32 `protobuf:"fixed32,15,opt,name=bound_right,json=boundRight,def=0.3" json:"bound_right,omitempty"`
	BoundTop                   *float32 `protobuf:"fixed32,16,opt,name=bound_top,json=boundTop,def=0.3" json:"bound_top,omitempty"`
	SaliencyWeight             *float32 `protobuf:"fixed32,3,opt,name=saliency_weight,json=saliencyWeight,def=20" json:"saliency_weight,omitempty"`
	ScaleWeightByFlowMagnitude *bool    `protobuf:"varint,8,opt,name=scale_weight_by_flow_magnitude,json=scaleWeightByFlowMagnitude,def=0" json:"scale_weight_by_flow_magnitude,omitempty"`
	MinFeatures                *int32   `protobuf:"varint,4,opt,name=min_features,json=minFeatures,def=5" json:"min_features,omitempty"`
	UseOnlyForegroundRegions   *bool    `protobuf:"varint,9,opt,name=use_only_foreground_regions,json=useOnlyForegroundRegions,def=0" json:"use_only_foreground_regions,omitempty"`
	MinIrlsModeWeight          *float32 `protobuf:"fixed32,10,opt,name=min_irls_mode_weight,json=minIrlsModeWeight,def=10" json:"min_irls_mode_weight,omitempty"`
	NumTopIrlsModes            *int32   `protobuf:"varint,11,opt,name=num_top_irls_modes,json=numTopIrlsModes,def=3" json:"num_top_irls_modes,omitempty"`
	ModeBandWidth              *float32 `protobuf:"fixed32,12,opt,name=mode_band_width,json=modeBandWidth,def=0.1" json:"mode_band_width,omitempty"`
	SelectionFrameRadius       *int32   `protobuf:"varint,5,opt,name=selection_frame_radius,json=selectionFrameRadius,def=5" json:"selection_frame_radius,omitempty"`
	SelectionSupportDistance   *float32 `protobuf:"fixed32,6,opt,name=selection_support_distance,json=selectionSupportDistance,def=0.2" json:"selection_support_distance,omitempty"`
	SelectionMinimumSupport    *int32   `protobuf:"varint,7,opt,name=selection_minimum_support,json=selectionMinimumSupport,def=4" json:"selection_minimum_support,omitempty"`
	FilteringSigmaSpace        *float32 `protobuf:"fixed32,13,opt,name=filtering_sigma_space,json=filteringSigmaSpace,def=0.05" json:"filtering_sigma_space,omitempty"`
	FilteringSigmaTime         *float32 `protobuf:"fixed32,14,opt,name=filtering_sigma_time,json=filteringSigmaTime,def=5" json:"filtering_sigma_time,omitempty"`
}

func (m *MotionSaliencyOptions) Reset()      { *m = MotionSaliencyOptions{} }
func (*MotionSaliencyOptions) ProtoMessage() {}
func (*MotionSaliencyOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_7efee98fc5c269ef, []int{0}
}
func (m *MotionSaliencyOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *MotionSaliencyOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_MotionSaliencyOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *MotionSaliencyOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MotionSaliencyOptions.Merge(m, src)
}
func (m *MotionSaliencyOptions) XXX_Size() int {
	return m.Size()
}
func (m *MotionSaliencyOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_MotionSaliencyOptions.DiscardUnknown(m)
}

var xxx_messageInfo_MotionSaliencyOptions proto.InternalMessageInfo

const Default_MotionSaliencyOptions_BoundLeft float32 = 0.3
const Default_MotionSaliencyOptions_BoundBottom float32 = 0.3
const Default_MotionSaliencyOptions_BoundRight float32 = 0.3
const Default_MotionSaliencyOptions_BoundTop float32 = 0.3
const Default_MotionSaliencyOptions_SaliencyWeight float32 = 20
const Default_MotionSaliencyOptions_ScaleWeightByFlowMagnitude bool = false
const Default_MotionSaliencyOptions_MinFeatures int32 = 5
const Default_MotionSaliencyOptions_UseOnlyForegroundRegions bool = false
const Default_MotionSaliencyOptions_MinIrlsModeWeight float32 = 10
const Default_MotionSaliencyOptions_NumTopIrlsModes int32 = 3
const Default_MotionSaliencyOptions_ModeBandWidth float32 = 0.1
const Default_MotionSaliencyOptions_SelectionFrameRadius int32 = 5
const Default_MotionSaliencyOptions_SelectionSupportDistance float32 = 0.2
const Default_MotionSaliencyOptions_SelectionMinimumSupport int32 = 4
const Default_MotionSaliencyOptions_FilteringSigmaSpace float32 = 0.05
const Default_MotionSaliencyOptions_FilteringSigmaTime float32 = 5

func (m *MotionSaliencyOptions) GetBoundLeft() float32 {
	if m != nil && m.BoundLeft != nil {
		return *m.BoundLeft
	}
	return Default_MotionSaliencyOptions_BoundLeft
}

func (m *MotionSaliencyOptions) GetBoundBottom() float32 {
	if m != nil && m.BoundBottom != nil {
		return *m.BoundBottom
	}
	return Default_MotionSaliencyOptions_BoundBottom
}

func (m *MotionSaliencyOptions) GetBoundRight() float32 {
	if m != nil && m.BoundRight != nil {
		return *m.BoundRight
	}
	return Default_MotionSaliencyOptions_BoundRight
}

func (m *MotionSaliencyOptions) GetBoundTop() float32 {
	if m != nil && m.BoundTop != nil {
		return *m.BoundTop
	}
	return Default_MotionSaliencyOptions_BoundTop
}

func (m *MotionSaliencyOptions) GetSaliencyWeight() float32 {
	if m != nil && m.SaliencyWeight != nil {
		return *m.SaliencyWeight
	}
	return Default_MotionSaliencyOptions_SaliencyWeight
}

func (m *MotionSaliencyOptions) GetScaleWeightByFlowMagnitude() bool {
	if m != nil && m.ScaleWeightByFlowMagnitude != nil {
		return *m.ScaleWeightByFlowMagnitude
	}
	return Default_MotionSaliencyOptions_ScaleWeightByFlowMagnitude
}

func (m *MotionSaliencyOptions) GetMinFeatures() int32 {
	if m != nil && m.MinFeatures != nil {
		return *m.MinFeatures
	}
	return Default_MotionSaliencyOptions_MinFeatures
}

func (m *MotionSaliencyOptions) GetUseOnlyForegroundRegions() bool {
	if m != nil && m.UseOnlyForegroundRegions != nil {
		return *m.UseOnlyForegroundRegions
	}
	return Default_MotionSaliencyOptions_UseOnlyForegroundRegions
}

func (m *MotionSaliencyOptions) GetMinIrlsModeWeight() float32 {
	if m != nil && m.MinIrlsModeWeight != nil {
		return *m.MinIrlsModeWeight
	}
	return Default_MotionSaliencyOptions_MinIrlsModeWeight
}

func (m *MotionSaliencyOptions) GetNumTopIrlsModes() int32 {
	if m != nil && m.NumTopIrlsModes != nil {
		return *m.NumTopIrlsModes
	}
	return Default_MotionSaliencyOptions_NumTopIrlsModes
}

func (m *MotionSaliencyOptions) GetModeBandWidth() float32 {
	if m != nil && m.ModeBandWidth != nil {
		return *m.ModeBandWidth
	}
	return Default_MotionSaliencyOptions_ModeBandWidth
}

func (m *MotionSaliencyOptions) GetSelectionFrameRadius() int32 {
	if m != nil && m.SelectionFrameRadius != nil {
		return *m.SelectionFrameRadius
	}
	return Default_MotionSaliencyOptions_SelectionFrameRadius
}

func (m *MotionSaliencyOptions) GetSelectionSupportDistance() float32 {
	if m != nil && m.SelectionSupportDistance != nil {
		return *m.SelectionSupportDistance
	}
	return Default_MotionSaliencyOptions_SelectionSupportDistance
}

func (m *MotionSaliencyOptions) GetSelectionMinimumSupport() int32 {
	if m != nil && m.SelectionMinimumSupport != nil {
		return *m.SelectionMinimumSupport
	}
	return Default_MotionSaliencyOptions_SelectionMinimumSupport
}

func (m *MotionSaliencyOptions) GetFilteringSigmaSpace() float32 {
	if m != nil && m.FilteringSigmaSpace != nil {
		return *m.FilteringSigmaSpace
	}
	return Default_MotionSaliencyOptions_FilteringSigmaSpace
}

func (m *MotionSaliencyOptions) GetFilteringSigmaTime() float32 {
	if m != nil && m.FilteringSigmaTime != nil {
		return *m.FilteringSigmaTime
	}
	return Default_MotionSaliencyOptions_FilteringSigmaTime
}

func init() {
	proto.RegisterType((*MotionSaliencyOptions)(nil), "mediapipe.MotionSaliencyOptions")
}

func init() {
	proto.RegisterFile("mediapipe/util/tracking/motion_saliency.proto", fileDescriptor_7efee98fc5c269ef)
}

var fileDescriptor_7efee98fc5c269ef = []byte{
	// 621 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x74, 0x93, 0xcb, 0x6a, 0x1b, 0x3f,
	0x14, 0xc6, 0x3d, 0xb9, 0xfd, 0x13, 0xe5, 0xf6, 0xaf, 0x9a, 0xb4, 0x6a, 0x0a, 0xc2, 0x84, 0x50,
	0x02, 0xa5, 0xb6, 0x13, 0x37, 0xb4, 0x04, 0xba, 0xa8, 0x09, 0x86, 0x40, 0x4d, 0x40, 0x0e, 0x04,
	0xba, 0x11, 0xb2, 0x47, 0x33, 0x11, 0xd5, 0x65, 0x18, 0x69, 0x30, 0xde, 0xf5, 0x11, 0xfa, 0x18,
	0x7d, 0x94, 0x42, 0x37, 0x59, 0x66, 0xd9, 0x4c, 0x36, 0x5d, 0xe6, 0x11, 0xca, 0x68, 0x3c, 0x9e,
	0xb6, 0xd0, 0xa5, 0xce, 0xf7, 0xfb, 0xce, 0xa7, 0x73, 0x66, 0x04, 0x5e, 0x29, 0x1e, 0x0a, 0x96,
	0x88, 0x84, 0xb7, 0x33, 0x27, 0x64, 0xdb, 0xa5, 0x6c, 0xfc, 0x49, 0xe8, 0xb8, 0xad, 0x8c, 0x13,
	0x46, 0x53, 0xcb, 0xa4, 0xe0, 0x7a, 0x3c, 0x6d, 0x25, 0xa9, 0x71, 0x06, 0xae, 0xcd, 0xf1, 0xfd,
	0xef, 0x2b, 0x60, 0x77, 0xe0, 0xa1, 0xe1, 0x8c, 0xb9, 0x48, 0x8a, 0x93, 0x85, 0xfb, 0x00, 0x8c,
	0x4c, 0xa6, 0x43, 0x2a, 0x79, 0xe4, 0x50, 0xd0, 0x0c, 0x0e, 0x17, 0x4e, 0x17, 0x3b, 0xad, 0x2e,
	0x59, 0xf3, 0xe5, 0x0f, 0x3c, 0x72, 0xf0, 0x05, 0xd8, 0x28, 0x99, 0x91, 0x71, 0xce, 0x28, 0xb4,
	0x50, 0x53, 0xeb, 0x5e, 0xe8, 0xf9, 0x3a, 0x3c, 0x00, 0xe5, 0x91, 0xa6, 0x22, 0xbe, 0x76, 0x68,
	0xbb, 0xc6, 0xca, 0x0c, 0x52, 0x94, 0x61, 0x13, 0x94, 0xad, 0xa9, 0x33, 0x09, 0xfa, 0xbf, 0x66,
	0x56, 0x7d, 0xf5, 0xd2, 0x24, 0xf0, 0x25, 0xd8, 0xae, 0x46, 0xa1, 0x13, 0xee, 0x7b, 0x2d, 0x7a,
	0x6e, 0xe1, 0xb8, 0x43, 0xb6, 0x2a, 0xe9, 0xca, 0x2b, 0xf0, 0x1c, 0x60, 0x3b, 0x66, 0x92, 0xcf,
	0x48, 0x3a, 0x9a, 0xd2, 0x48, 0x9a, 0x09, 0x55, 0x2c, 0xd6, 0xc2, 0x65, 0x21, 0x47, 0xab, 0xcd,
	0xe0, 0x70, 0xf5, 0x74, 0x39, 0x62, 0xd2, 0x72, 0xb2, 0xe7, 0xe1, 0xd2, 0xdb, 0x9b, 0xf6, 0xa5,
	0x99, 0x0c, 0x2a, 0x10, 0x1e, 0x80, 0x0d, 0x25, 0x34, 0x8d, 0x38, 0x73, 0x59, 0xca, 0x2d, 0x5a,
	0x6a, 0x06, 0x87, 0xcb, 0xa7, 0xc1, 0x09, 0x59, 0x57, 0x42, 0xf7, 0x67, 0x55, 0x78, 0x06, 0x9e,
	0x67, 0x96, 0x53, 0xa3, 0xe5, 0x94, 0x46, 0x26, 0xe5, 0x71, 0x5a, 0xce, 0xcc, 0xe3, 0x62, 0xa1,
	0x68, 0xed, 0xf7, 0x34, 0x94, 0x59, 0x7e, 0xa1, 0xe5, 0xb4, 0x3f, 0xe7, 0x48, 0x89, 0xc1, 0x2e,
	0xd8, 0x29, 0xb2, 0x44, 0x2a, 0x2d, 0x55, 0x26, 0xac, 0xae, 0x8f, 0x40, 0x39, 0xe8, 0x51, 0x87,
	0x3c, 0x52, 0x42, 0x9f, 0xa7, 0xd2, 0x0e, 0x4c, 0x38, 0xbb, 0x2f, 0x6c, 0x01, 0xa8, 0x33, 0x55,
	0x2c, 0xae, 0x36, 0x5a, 0xb4, 0x5e, 0x5e, 0xb3, 0x4b, 0xb6, 0x75, 0xa6, 0x2e, 0x4d, 0x52, 0x99,
	0x6c, 0xb1, 0x48, 0xdf, 0x7b, 0xc4, 0x74, 0x48, 0x27, 0x22, 0x74, 0xd7, 0x68, 0xa3, 0x5a, 0xf8,
	0x11, 0xd9, 0x2c, 0xb4, 0x1e, 0xd3, 0xe1, 0x55, 0xa1, 0xc0, 0x37, 0xe0, 0x89, 0xe5, 0x92, 0x8f,
	0xfd, 0xaf, 0x14, 0xa5, 0x4c, 0x71, 0x9a, 0xb2, 0x50, 0x64, 0x16, 0x2d, 0x57, 0x7b, 0xd8, 0x99,
	0x03, 0xfd, 0x42, 0x27, 0x5e, 0x86, 0xef, 0xc1, 0x5e, 0x6d, 0xb4, 0x59, 0x92, 0x98, 0xd4, 0xd1,
	0x50, 0x58, 0xc7, 0xf4, 0x98, 0xa3, 0x95, 0x2a, 0xf0, 0x98, 0xa0, 0x39, 0x36, 0x2c, 0xa9, 0xb3,
	0x19, 0x04, 0xdf, 0x81, 0x67, 0x75, 0x0b, 0x25, 0xb4, 0x50, 0x99, 0xaa, 0x5a, 0xa1, 0xff, 0xca,
	0xf8, 0xd7, 0xe4, 0xe9, 0x9c, 0x19, 0x94, 0xc8, 0xac, 0x0d, 0x7c, 0x0b, 0x76, 0x23, 0x21, 0x1d,
	0x4f, 0x85, 0x8e, 0xa9, 0x15, 0xb1, 0x62, 0xd4, 0x26, 0x6c, 0xcc, 0xd1, 0xa6, 0x0f, 0x5f, 0xea,
	0xb4, 0x3a, 0x27, 0xe4, 0xf1, 0x1c, 0x19, 0x16, 0xc4, 0xb0, 0x00, 0x8a, 0xcf, 0xf0, 0xb7, 0xd3,
	0x09, 0xc5, 0xd1, 0x96, 0x37, 0x06, 0x27, 0x04, 0xfe, 0xe9, 0xba, 0x14, 0x8a, 0xf7, 0xc4, 0xcd,
	0x1d, 0x6e, 0xdc, 0xde, 0xe1, 0xc6, 0xc3, 0x1d, 0x0e, 0x3e, 0xe7, 0x38, 0xf8, 0x9a, 0xe3, 0xe0,
	0x5b, 0x8e, 0x83, 0x9b, 0x1c, 0x07, 0x3f, 0x72, 0x1c, 0xfc, 0xcc, 0x71, 0xe3, 0x21, 0xc7, 0xc1,
	0x97, 0x7b, 0xdc, 0xb8, 0xb9, 0xc7, 0x8d, 0xdb, 0x7b, 0xdc, 0xf8, 0xd8, 0x8d, 0x85, 0xbb, 0xce,
	0x46, 0xad, 0xb1, 0x51, 0xed, 0xd8, 0x98, 0x58, 0xf2, 0x76, 0xfd, 0x9e, 0xff, 0xf1, 0xb2, 0x7f,
	0x05, 0x00, 0x00, 0xff, 0xff, 0x74, 0x9e, 0x83, 0x00, 0xf3, 0x03, 0x00, 0x00,
}

func (this *MotionSaliencyOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*MotionSaliencyOptions)
	if !ok {
		that2, ok := that.(MotionSaliencyOptions)
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
	if this.BoundLeft != nil && that1.BoundLeft != nil {
		if *this.BoundLeft != *that1.BoundLeft {
			return false
		}
	} else if this.BoundLeft != nil {
		return false
	} else if that1.BoundLeft != nil {
		return false
	}
	if this.BoundBottom != nil && that1.BoundBottom != nil {
		if *this.BoundBottom != *that1.BoundBottom {
			return false
		}
	} else if this.BoundBottom != nil {
		return false
	} else if that1.BoundBottom != nil {
		return false
	}
	if this.BoundRight != nil && that1.BoundRight != nil {
		if *this.BoundRight != *that1.BoundRight {
			return false
		}
	} else if this.BoundRight != nil {
		return false
	} else if that1.BoundRight != nil {
		return false
	}
	if this.BoundTop != nil && that1.BoundTop != nil {
		if *this.BoundTop != *that1.BoundTop {
			return false
		}
	} else if this.BoundTop != nil {
		return false
	} else if that1.BoundTop != nil {
		return false
	}
	if this.SaliencyWeight != nil && that1.SaliencyWeight != nil {
		if *this.SaliencyWeight != *that1.SaliencyWeight {
			return false
		}
	} else if this.SaliencyWeight != nil {
		return false
	} else if that1.SaliencyWeight != nil {
		return false
	}
	if this.ScaleWeightByFlowMagnitude != nil && that1.ScaleWeightByFlowMagnitude != nil {
		if *this.ScaleWeightByFlowMagnitude != *that1.ScaleWeightByFlowMagnitude {
			return false
		}
	} else if this.ScaleWeightByFlowMagnitude != nil {
		return false
	} else if that1.ScaleWeightByFlowMagnitude != nil {
		return false
	}
	if this.MinFeatures != nil && that1.MinFeatures != nil {
		if *this.MinFeatures != *that1.MinFeatures {
			return false
		}
	} else if this.MinFeatures != nil {
		return false
	} else if that1.MinFeatures != nil {
		return false
	}
	if this.UseOnlyForegroundRegions != nil && that1.UseOnlyForegroundRegions != nil {
		if *this.UseOnlyForegroundRegions != *that1.UseOnlyForegroundRegions {
			return false
		}
	} else if this.UseOnlyForegroundRegions != nil {
		return false
	} else if that1.UseOnlyForegroundRegions != nil {
		return false
	}
	if this.MinIrlsModeWeight != nil && that1.MinIrlsModeWeight != nil {
		if *this.MinIrlsModeWeight != *that1.MinIrlsModeWeight {
			return false
		}
	} else if this.MinIrlsModeWeight != nil {
		return false
	} else if that1.MinIrlsModeWeight != nil {
		return false
	}
	if this.NumTopIrlsModes != nil && that1.NumTopIrlsModes != nil {
		if *this.NumTopIrlsModes != *that1.NumTopIrlsModes {
			return false
		}
	} else if this.NumTopIrlsModes != nil {
		return false
	} else if that1.NumTopIrlsModes != nil {
		return false
	}
	if this.ModeBandWidth != nil && that1.ModeBandWidth != nil {
		if *this.ModeBandWidth != *that1.ModeBandWidth {
			return false
		}
	} else if this.ModeBandWidth != nil {
		return false
	} else if that1.ModeBandWidth != nil {
		return false
	}
	if this.SelectionFrameRadius != nil && that1.SelectionFrameRadius != nil {
		if *this.SelectionFrameRadius != *that1.SelectionFrameRadius {
			return false
		}
	} else if this.SelectionFrameRadius != nil {
		return false
	} else if that1.SelectionFrameRadius != nil {
		return false
	}
	if this.SelectionSupportDistance != nil && that1.SelectionSupportDistance != nil {
		if *this.SelectionSupportDistance != *that1.SelectionSupportDistance {
			return false
		}
	} else if this.SelectionSupportDistance != nil {
		return false
	} else if that1.SelectionSupportDistance != nil {
		return false
	}
	if this.SelectionMinimumSupport != nil && that1.SelectionMinimumSupport != nil {
		if *this.SelectionMinimumSupport != *that1.SelectionMinimumSupport {
			return false
		}
	} else if this.SelectionMinimumSupport != nil {
		return false
	} else if that1.SelectionMinimumSupport != nil {
		return false
	}
	if this.FilteringSigmaSpace != nil && that1.FilteringSigmaSpace != nil {
		if *this.FilteringSigmaSpace != *that1.FilteringSigmaSpace {
			return false
		}
	} else if this.FilteringSigmaSpace != nil {
		return false
	} else if that1.FilteringSigmaSpace != nil {
		return false
	}
	if this.FilteringSigmaTime != nil && that1.FilteringSigmaTime != nil {
		if *this.FilteringSigmaTime != *that1.FilteringSigmaTime {
			return false
		}
	} else if this.FilteringSigmaTime != nil {
		return false
	} else if that1.FilteringSigmaTime != nil {
		return false
	}
	return true
}
func (this *MotionSaliencyOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 20)
	s = append(s, "&tracking.MotionSaliencyOptions{")
	if this.BoundLeft != nil {
		s = append(s, "BoundLeft: "+valueToGoStringMotionSaliency(this.BoundLeft, "float32")+",\n")
	}
	if this.BoundBottom != nil {
		s = append(s, "BoundBottom: "+valueToGoStringMotionSaliency(this.BoundBottom, "float32")+",\n")
	}
	if this.BoundRight != nil {
		s = append(s, "BoundRight: "+valueToGoStringMotionSaliency(this.BoundRight, "float32")+",\n")
	}
	if this.BoundTop != nil {
		s = append(s, "BoundTop: "+valueToGoStringMotionSaliency(this.BoundTop, "float32")+",\n")
	}
	if this.SaliencyWeight != nil {
		s = append(s, "SaliencyWeight: "+valueToGoStringMotionSaliency(this.SaliencyWeight, "float32")+",\n")
	}
	if this.ScaleWeightByFlowMagnitude != nil {
		s = append(s, "ScaleWeightByFlowMagnitude: "+valueToGoStringMotionSaliency(this.ScaleWeightByFlowMagnitude, "bool")+",\n")
	}
	if this.MinFeatures != nil {
		s = append(s, "MinFeatures: "+valueToGoStringMotionSaliency(this.MinFeatures, "int32")+",\n")
	}
	if this.UseOnlyForegroundRegions != nil {
		s = append(s, "UseOnlyForegroundRegions: "+valueToGoStringMotionSaliency(this.UseOnlyForegroundRegions, "bool")+",\n")
	}
	if this.MinIrlsModeWeight != nil {
		s = append(s, "MinIrlsModeWeight: "+valueToGoStringMotionSaliency(this.MinIrlsModeWeight, "float32")+",\n")
	}
	if this.NumTopIrlsModes != nil {
		s = append(s, "NumTopIrlsModes: "+valueToGoStringMotionSaliency(this.NumTopIrlsModes, "int32")+",\n")
	}
	if this.ModeBandWidth != nil {
		s = append(s, "ModeBandWidth: "+valueToGoStringMotionSaliency(this.ModeBandWidth, "float32")+",\n")
	}
	if this.SelectionFrameRadius != nil {
		s = append(s, "SelectionFrameRadius: "+valueToGoStringMotionSaliency(this.SelectionFrameRadius, "int32")+",\n")
	}
	if this.SelectionSupportDistance != nil {
		s = append(s, "SelectionSupportDistance: "+valueToGoStringMotionSaliency(this.SelectionSupportDistance, "float32")+",\n")
	}
	if this.SelectionMinimumSupport != nil {
		s = append(s, "SelectionMinimumSupport: "+valueToGoStringMotionSaliency(this.SelectionMinimumSupport, "int32")+",\n")
	}
	if this.FilteringSigmaSpace != nil {
		s = append(s, "FilteringSigmaSpace: "+valueToGoStringMotionSaliency(this.FilteringSigmaSpace, "float32")+",\n")
	}
	if this.FilteringSigmaTime != nil {
		s = append(s, "FilteringSigmaTime: "+valueToGoStringMotionSaliency(this.FilteringSigmaTime, "float32")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringMotionSaliency(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *MotionSaliencyOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *MotionSaliencyOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *MotionSaliencyOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.BoundTop != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.BoundTop))))
		i--
		dAtA[i] = 0x1
		i--
		dAtA[i] = 0x85
	}
	if m.BoundRight != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.BoundRight))))
		i--
		dAtA[i] = 0x7d
	}
	if m.FilteringSigmaTime != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.FilteringSigmaTime))))
		i--
		dAtA[i] = 0x75
	}
	if m.FilteringSigmaSpace != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.FilteringSigmaSpace))))
		i--
		dAtA[i] = 0x6d
	}
	if m.ModeBandWidth != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.ModeBandWidth))))
		i--
		dAtA[i] = 0x65
	}
	if m.NumTopIrlsModes != nil {
		i = encodeVarintMotionSaliency(dAtA, i, uint64(*m.NumTopIrlsModes))
		i--
		dAtA[i] = 0x58
	}
	if m.MinIrlsModeWeight != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.MinIrlsModeWeight))))
		i--
		dAtA[i] = 0x55
	}
	if m.UseOnlyForegroundRegions != nil {
		i--
		if *m.UseOnlyForegroundRegions {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x48
	}
	if m.ScaleWeightByFlowMagnitude != nil {
		i--
		if *m.ScaleWeightByFlowMagnitude {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x40
	}
	if m.SelectionMinimumSupport != nil {
		i = encodeVarintMotionSaliency(dAtA, i, uint64(*m.SelectionMinimumSupport))
		i--
		dAtA[i] = 0x38
	}
	if m.SelectionSupportDistance != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.SelectionSupportDistance))))
		i--
		dAtA[i] = 0x35
	}
	if m.SelectionFrameRadius != nil {
		i = encodeVarintMotionSaliency(dAtA, i, uint64(*m.SelectionFrameRadius))
		i--
		dAtA[i] = 0x28
	}
	if m.MinFeatures != nil {
		i = encodeVarintMotionSaliency(dAtA, i, uint64(*m.MinFeatures))
		i--
		dAtA[i] = 0x20
	}
	if m.SaliencyWeight != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.SaliencyWeight))))
		i--
		dAtA[i] = 0x1d
	}
	if m.BoundBottom != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.BoundBottom))))
		i--
		dAtA[i] = 0x15
	}
	if m.BoundLeft != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.BoundLeft))))
		i--
		dAtA[i] = 0xd
	}
	return len(dAtA) - i, nil
}

func encodeVarintMotionSaliency(dAtA []byte, offset int, v uint64) int {
	offset -= sovMotionSaliency(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *MotionSaliencyOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.BoundLeft != nil {
		n += 5
	}
	if m.BoundBottom != nil {
		n += 5
	}
	if m.SaliencyWeight != nil {
		n += 5
	}
	if m.MinFeatures != nil {
		n += 1 + sovMotionSaliency(uint64(*m.MinFeatures))
	}
	if m.SelectionFrameRadius != nil {
		n += 1 + sovMotionSaliency(uint64(*m.SelectionFrameRadius))
	}
	if m.SelectionSupportDistance != nil {
		n += 5
	}
	if m.SelectionMinimumSupport != nil {
		n += 1 + sovMotionSaliency(uint64(*m.SelectionMinimumSupport))
	}
	if m.ScaleWeightByFlowMagnitude != nil {
		n += 2
	}
	if m.UseOnlyForegroundRegions != nil {
		n += 2
	}
	if m.MinIrlsModeWeight != nil {
		n += 5
	}
	if m.NumTopIrlsModes != nil {
		n += 1 + sovMotionSaliency(uint64(*m.NumTopIrlsModes))
	}
	if m.ModeBandWidth != nil {
		n += 5
	}
	if m.FilteringSigmaSpace != nil {
		n += 5
	}
	if m.FilteringSigmaTime != nil {
		n += 5
	}
	if m.BoundRight != nil {
		n += 5
	}
	if m.BoundTop != nil {
		n += 6
	}
	return n
}

func sovMotionSaliency(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozMotionSaliency(x uint64) (n int) {
	return sovMotionSaliency(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *MotionSaliencyOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&MotionSaliencyOptions{`,
		`BoundLeft:` + valueToStringMotionSaliency(this.BoundLeft) + `,`,
		`BoundBottom:` + valueToStringMotionSaliency(this.BoundBottom) + `,`,
		`SaliencyWeight:` + valueToStringMotionSaliency(this.SaliencyWeight) + `,`,
		`MinFeatures:` + valueToStringMotionSaliency(this.MinFeatures) + `,`,
		`SelectionFrameRadius:` + valueToStringMotionSaliency(this.SelectionFrameRadius) + `,`,
		`SelectionSupportDistance:` + valueToStringMotionSaliency(this.SelectionSupportDistance) + `,`,
		`SelectionMinimumSupport:` + valueToStringMotionSaliency(this.SelectionMinimumSupport) + `,`,
		`ScaleWeightByFlowMagnitude:` + valueToStringMotionSaliency(this.ScaleWeightByFlowMagnitude) + `,`,
		`UseOnlyForegroundRegions:` + valueToStringMotionSaliency(this.UseOnlyForegroundRegions) + `,`,
		`MinIrlsModeWeight:` + valueToStringMotionSaliency(this.MinIrlsModeWeight) + `,`,
		`NumTopIrlsModes:` + valueToStringMotionSaliency(this.NumTopIrlsModes) + `,`,
		`ModeBandWidth:` + valueToStringMotionSaliency(this.ModeBandWidth) + `,`,
		`FilteringSigmaSpace:` + valueToStringMotionSaliency(this.FilteringSigmaSpace) + `,`,
		`FilteringSigmaTime:` + valueToStringMotionSaliency(this.FilteringSigmaTime) + `,`,
		`BoundRight:` + valueToStringMotionSaliency(this.BoundRight) + `,`,
		`BoundTop:` + valueToStringMotionSaliency(this.BoundTop) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringMotionSaliency(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *MotionSaliencyOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMotionSaliency
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
			return fmt.Errorf("proto: MotionSaliencyOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: MotionSaliencyOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field BoundLeft", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.BoundLeft = &v2
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field BoundBottom", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.BoundBottom = &v2
		case 3:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field SaliencyWeight", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.SaliencyWeight = &v2
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field MinFeatures", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMotionSaliency
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
			m.MinFeatures = &v
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field SelectionFrameRadius", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMotionSaliency
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
			m.SelectionFrameRadius = &v
		case 6:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field SelectionSupportDistance", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.SelectionSupportDistance = &v2
		case 7:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field SelectionMinimumSupport", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMotionSaliency
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
			m.SelectionMinimumSupport = &v
		case 8:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field ScaleWeightByFlowMagnitude", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMotionSaliency
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
			m.ScaleWeightByFlowMagnitude = &b
		case 9:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field UseOnlyForegroundRegions", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMotionSaliency
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
			m.UseOnlyForegroundRegions = &b
		case 10:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field MinIrlsModeWeight", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.MinIrlsModeWeight = &v2
		case 11:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NumTopIrlsModes", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMotionSaliency
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
			m.NumTopIrlsModes = &v
		case 12:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModeBandWidth", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.ModeBandWidth = &v2
		case 13:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field FilteringSigmaSpace", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.FilteringSigmaSpace = &v2
		case 14:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field FilteringSigmaTime", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.FilteringSigmaTime = &v2
		case 15:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field BoundRight", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.BoundRight = &v2
		case 16:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field BoundTop", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.BoundTop = &v2
		default:
			iNdEx = preIndex
			skippy, err := skipMotionSaliency(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMotionSaliency
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
func skipMotionSaliency(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowMotionSaliency
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
					return 0, ErrIntOverflowMotionSaliency
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
					return 0, ErrIntOverflowMotionSaliency
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
				return 0, ErrInvalidLengthMotionSaliency
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupMotionSaliency
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthMotionSaliency
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthMotionSaliency        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowMotionSaliency          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupMotionSaliency = fmt.Errorf("proto: unexpected end of group")
)
