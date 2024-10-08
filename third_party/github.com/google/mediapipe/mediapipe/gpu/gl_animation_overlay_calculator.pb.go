// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/gpu/gl_animation_overlay_calculator.proto

package gpu

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

type GlAnimationOverlayCalculatorOptions struct {
	AspectRatio        *float32 `protobuf:"fixed32,1,opt,name=aspect_ratio,json=aspectRatio,def=0.75" json:"aspect_ratio,omitempty"`
	VerticalFovDegrees *float32 `protobuf:"fixed32,2,opt,name=vertical_fov_degrees,json=verticalFovDegrees,def=70" json:"vertical_fov_degrees,omitempty"`
	ZClippingPlaneNear *float32 `protobuf:"fixed32,3,opt,name=z_clipping_plane_near,json=zClippingPlaneNear,def=0.1" json:"z_clipping_plane_near,omitempty"`
	ZClippingPlaneFar  *float32 `protobuf:"fixed32,4,opt,name=z_clipping_plane_far,json=zClippingPlaneFar,def=1000" json:"z_clipping_plane_far,omitempty"`
	AnimationSpeedFps  *float32 `protobuf:"fixed32,5,opt,name=animation_speed_fps,json=animationSpeedFps,def=25" json:"animation_speed_fps,omitempty"`
}

func (m *GlAnimationOverlayCalculatorOptions) Reset()      { *m = GlAnimationOverlayCalculatorOptions{} }
func (*GlAnimationOverlayCalculatorOptions) ProtoMessage() {}
func (*GlAnimationOverlayCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_fd83f46579f7d1fb, []int{0}
}
func (m *GlAnimationOverlayCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *GlAnimationOverlayCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_GlAnimationOverlayCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *GlAnimationOverlayCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GlAnimationOverlayCalculatorOptions.Merge(m, src)
}
func (m *GlAnimationOverlayCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *GlAnimationOverlayCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_GlAnimationOverlayCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_GlAnimationOverlayCalculatorOptions proto.InternalMessageInfo

const Default_GlAnimationOverlayCalculatorOptions_AspectRatio float32 = 0.75
const Default_GlAnimationOverlayCalculatorOptions_VerticalFovDegrees float32 = 70
const Default_GlAnimationOverlayCalculatorOptions_ZClippingPlaneNear float32 = 0.1
const Default_GlAnimationOverlayCalculatorOptions_ZClippingPlaneFar float32 = 1000
const Default_GlAnimationOverlayCalculatorOptions_AnimationSpeedFps float32 = 25

func (m *GlAnimationOverlayCalculatorOptions) GetAspectRatio() float32 {
	if m != nil && m.AspectRatio != nil {
		return *m.AspectRatio
	}
	return Default_GlAnimationOverlayCalculatorOptions_AspectRatio
}

func (m *GlAnimationOverlayCalculatorOptions) GetVerticalFovDegrees() float32 {
	if m != nil && m.VerticalFovDegrees != nil {
		return *m.VerticalFovDegrees
	}
	return Default_GlAnimationOverlayCalculatorOptions_VerticalFovDegrees
}

func (m *GlAnimationOverlayCalculatorOptions) GetZClippingPlaneNear() float32 {
	if m != nil && m.ZClippingPlaneNear != nil {
		return *m.ZClippingPlaneNear
	}
	return Default_GlAnimationOverlayCalculatorOptions_ZClippingPlaneNear
}

func (m *GlAnimationOverlayCalculatorOptions) GetZClippingPlaneFar() float32 {
	if m != nil && m.ZClippingPlaneFar != nil {
		return *m.ZClippingPlaneFar
	}
	return Default_GlAnimationOverlayCalculatorOptions_ZClippingPlaneFar
}

func (m *GlAnimationOverlayCalculatorOptions) GetAnimationSpeedFps() float32 {
	if m != nil && m.AnimationSpeedFps != nil {
		return *m.AnimationSpeedFps
	}
	return Default_GlAnimationOverlayCalculatorOptions_AnimationSpeedFps
}

var E_GlAnimationOverlayCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*GlAnimationOverlayCalculatorOptions)(nil),
	Field:         174760573,
	Name:          "mediapipe.GlAnimationOverlayCalculatorOptions.ext",
	Tag:           "bytes,174760573,opt,name=ext",
	Filename:      "mediapipe/gpu/gl_animation_overlay_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_GlAnimationOverlayCalculatorOptions_Ext)
	proto.RegisterType((*GlAnimationOverlayCalculatorOptions)(nil), "mediapipe.GlAnimationOverlayCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/gpu/gl_animation_overlay_calculator.proto", fileDescriptor_fd83f46579f7d1fb)
}

var fileDescriptor_fd83f46579f7d1fb = []byte{
	// 396 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x92, 0xb1, 0xae, 0xd3, 0x30,
	0x18, 0x85, 0xe3, 0xe6, 0x32, 0xe0, 0xcb, 0x72, 0xcd, 0x45, 0x8a, 0x10, 0xb2, 0x2a, 0x40, 0xa2,
	0x2c, 0x49, 0x1a, 0x28, 0x95, 0xd8, 0xa0, 0xa8, 0x6c, 0x14, 0xa5, 0x1b, 0x8b, 0x65, 0xd2, 0x3f,
	0xc1, 0xc2, 0x89, 0x2d, 0x27, 0x0d, 0xd0, 0x89, 0x99, 0x89, 0xc7, 0x40, 0x3c, 0x07, 0x03, 0x63,
	0xc7, 0x8e, 0x34, 0x5d, 0x18, 0xfb, 0x02, 0x48, 0x28, 0x0d, 0x4d, 0x29, 0x5d, 0x58, 0xf3, 0x9d,
	0xef, 0xe8, 0xd7, 0x89, 0xf1, 0x83, 0x14, 0x66, 0x82, 0x6b, 0xa1, 0xc1, 0x4b, 0xf4, 0xdc, 0x4b,
	0x24, 0xe3, 0x99, 0x48, 0x79, 0x21, 0x54, 0xc6, 0x54, 0x09, 0x46, 0xf2, 0x0f, 0x2c, 0xe2, 0x32,
	0x9a, 0x4b, 0x5e, 0x28, 0xe3, 0x6a, 0xa3, 0x0a, 0x45, 0xae, 0xb6, 0xd2, 0xcd, 0xbb, 0x07, 0x3f,
	0x36, 0x3c, 0x85, 0x77, 0xca, 0xbc, 0xf5, 0xfe, 0x15, 0x6e, 0x7f, 0xb2, 0xf1, 0x9d, 0xe7, 0xf2,
	0xc9, 0xbe, 0x79, 0xd2, 0x14, 0x8f, 0xda, 0xd8, 0x44, 0xd7, 0x9f, 0x73, 0x72, 0x0f, 0x5f, 0xe3,
	0xb9, 0x86, 0xa8, 0x60, 0xa6, 0x0e, 0x3a, 0xa8, 0x8b, 0x7a, 0x9d, 0xc7, 0x67, 0xbe, 0x3b, 0x1c,
	0x84, 0xe7, 0x0d, 0x09, 0x6b, 0x40, 0x1e, 0xe2, 0xcb, 0x12, 0x4c, 0x21, 0x22, 0x2e, 0x59, 0xac,
	0x4a, 0x36, 0x83, 0xc4, 0x00, 0xe4, 0x4e, 0x67, 0x27, 0x74, 0x86, 0x7e, 0x48, 0xf6, 0x7c, 0xac,
	0xca, 0x67, 0x0d, 0x25, 0x8f, 0xf0, 0x8d, 0x05, 0x8b, 0xa4, 0xd0, 0x5a, 0x64, 0x09, 0xd3, 0x92,
	0x67, 0xc0, 0x32, 0xe0, 0xc6, 0xb1, 0x77, 0x9a, 0xed, 0xbb, 0xfd, 0x90, 0x2c, 0x46, 0x7f, 0x02,
	0x2f, 0x6b, 0xfe, 0x02, 0xb8, 0x21, 0x03, 0x7c, 0x79, 0xe2, 0xc5, 0xdc, 0x38, 0x67, 0xcd, 0x79,
	0x7d, 0xdf, 0xf7, 0xc3, 0x8b, 0x63, 0x6f, 0xcc, 0x0d, 0x09, 0xf0, 0xf5, 0xc3, 0x98, 0xb9, 0x06,
	0x98, 0xb1, 0x58, 0xe7, 0xce, 0x95, 0xe6, 0xc6, 0x60, 0x10, 0x5e, 0xb4, 0x78, 0x5a, 0xd3, 0xb1,
	0xce, 0x03, 0x8e, 0x6d, 0x78, 0x5f, 0x90, 0x5b, 0x6e, 0xbb, 0xab, 0x7b, 0x32, 0x93, 0xf3, 0xeb,
	0xdb, 0xd7, 0x69, 0x17, 0xf5, 0xce, 0x03, 0xf7, 0xaf, 0xd8, 0x7f, 0xec, 0x1b, 0xd6, 0xdd, 0x4f,
	0xd9, 0x72, 0x4d, 0xad, 0xd5, 0x9a, 0x5a, 0xdb, 0x35, 0x45, 0x1f, 0x2b, 0x8a, 0xbe, 0x54, 0x14,
	0x7d, 0xaf, 0x28, 0x5a, 0x56, 0x14, 0xfd, 0xa8, 0x28, 0xfa, 0x59, 0x51, 0x6b, 0x5b, 0x51, 0xf4,
	0x79, 0x43, 0xad, 0xe5, 0x86, 0x5a, 0xab, 0x0d, 0xb5, 0x5e, 0xdd, 0x4f, 0x44, 0xf1, 0x66, 0xfe,
	0xda, 0x8d, 0x54, 0xea, 0x25, 0x4a, 0x25, 0x12, 0xbc, 0xc3, 0xaf, 0x3f, 0x7a, 0x44, 0xbf, 0x03,
	0x00, 0x00, 0xff, 0xff, 0xad, 0xe3, 0xc3, 0xe3, 0x54, 0x02, 0x00, 0x00,
}

func (this *GlAnimationOverlayCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*GlAnimationOverlayCalculatorOptions)
	if !ok {
		that2, ok := that.(GlAnimationOverlayCalculatorOptions)
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
	if this.AspectRatio != nil && that1.AspectRatio != nil {
		if *this.AspectRatio != *that1.AspectRatio {
			return false
		}
	} else if this.AspectRatio != nil {
		return false
	} else if that1.AspectRatio != nil {
		return false
	}
	if this.VerticalFovDegrees != nil && that1.VerticalFovDegrees != nil {
		if *this.VerticalFovDegrees != *that1.VerticalFovDegrees {
			return false
		}
	} else if this.VerticalFovDegrees != nil {
		return false
	} else if that1.VerticalFovDegrees != nil {
		return false
	}
	if this.ZClippingPlaneNear != nil && that1.ZClippingPlaneNear != nil {
		if *this.ZClippingPlaneNear != *that1.ZClippingPlaneNear {
			return false
		}
	} else if this.ZClippingPlaneNear != nil {
		return false
	} else if that1.ZClippingPlaneNear != nil {
		return false
	}
	if this.ZClippingPlaneFar != nil && that1.ZClippingPlaneFar != nil {
		if *this.ZClippingPlaneFar != *that1.ZClippingPlaneFar {
			return false
		}
	} else if this.ZClippingPlaneFar != nil {
		return false
	} else if that1.ZClippingPlaneFar != nil {
		return false
	}
	if this.AnimationSpeedFps != nil && that1.AnimationSpeedFps != nil {
		if *this.AnimationSpeedFps != *that1.AnimationSpeedFps {
			return false
		}
	} else if this.AnimationSpeedFps != nil {
		return false
	} else if that1.AnimationSpeedFps != nil {
		return false
	}
	return true
}
func (this *GlAnimationOverlayCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 9)
	s = append(s, "&gpu.GlAnimationOverlayCalculatorOptions{")
	if this.AspectRatio != nil {
		s = append(s, "AspectRatio: "+valueToGoStringGlAnimationOverlayCalculator(this.AspectRatio, "float32")+",\n")
	}
	if this.VerticalFovDegrees != nil {
		s = append(s, "VerticalFovDegrees: "+valueToGoStringGlAnimationOverlayCalculator(this.VerticalFovDegrees, "float32")+",\n")
	}
	if this.ZClippingPlaneNear != nil {
		s = append(s, "ZClippingPlaneNear: "+valueToGoStringGlAnimationOverlayCalculator(this.ZClippingPlaneNear, "float32")+",\n")
	}
	if this.ZClippingPlaneFar != nil {
		s = append(s, "ZClippingPlaneFar: "+valueToGoStringGlAnimationOverlayCalculator(this.ZClippingPlaneFar, "float32")+",\n")
	}
	if this.AnimationSpeedFps != nil {
		s = append(s, "AnimationSpeedFps: "+valueToGoStringGlAnimationOverlayCalculator(this.AnimationSpeedFps, "float32")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringGlAnimationOverlayCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *GlAnimationOverlayCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *GlAnimationOverlayCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *GlAnimationOverlayCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.AnimationSpeedFps != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.AnimationSpeedFps))))
		i--
		dAtA[i] = 0x2d
	}
	if m.ZClippingPlaneFar != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.ZClippingPlaneFar))))
		i--
		dAtA[i] = 0x25
	}
	if m.ZClippingPlaneNear != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.ZClippingPlaneNear))))
		i--
		dAtA[i] = 0x1d
	}
	if m.VerticalFovDegrees != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.VerticalFovDegrees))))
		i--
		dAtA[i] = 0x15
	}
	if m.AspectRatio != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.AspectRatio))))
		i--
		dAtA[i] = 0xd
	}
	return len(dAtA) - i, nil
}

func encodeVarintGlAnimationOverlayCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovGlAnimationOverlayCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *GlAnimationOverlayCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.AspectRatio != nil {
		n += 5
	}
	if m.VerticalFovDegrees != nil {
		n += 5
	}
	if m.ZClippingPlaneNear != nil {
		n += 5
	}
	if m.ZClippingPlaneFar != nil {
		n += 5
	}
	if m.AnimationSpeedFps != nil {
		n += 5
	}
	return n
}

func sovGlAnimationOverlayCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozGlAnimationOverlayCalculator(x uint64) (n int) {
	return sovGlAnimationOverlayCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *GlAnimationOverlayCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&GlAnimationOverlayCalculatorOptions{`,
		`AspectRatio:` + valueToStringGlAnimationOverlayCalculator(this.AspectRatio) + `,`,
		`VerticalFovDegrees:` + valueToStringGlAnimationOverlayCalculator(this.VerticalFovDegrees) + `,`,
		`ZClippingPlaneNear:` + valueToStringGlAnimationOverlayCalculator(this.ZClippingPlaneNear) + `,`,
		`ZClippingPlaneFar:` + valueToStringGlAnimationOverlayCalculator(this.ZClippingPlaneFar) + `,`,
		`AnimationSpeedFps:` + valueToStringGlAnimationOverlayCalculator(this.AnimationSpeedFps) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringGlAnimationOverlayCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *GlAnimationOverlayCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowGlAnimationOverlayCalculator
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
			return fmt.Errorf("proto: GlAnimationOverlayCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: GlAnimationOverlayCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field AspectRatio", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.AspectRatio = &v2
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field VerticalFovDegrees", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.VerticalFovDegrees = &v2
		case 3:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field ZClippingPlaneNear", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.ZClippingPlaneNear = &v2
		case 4:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field ZClippingPlaneFar", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.ZClippingPlaneFar = &v2
		case 5:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field AnimationSpeedFps", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.AnimationSpeedFps = &v2
		default:
			iNdEx = preIndex
			skippy, err := skipGlAnimationOverlayCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthGlAnimationOverlayCalculator
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
func skipGlAnimationOverlayCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowGlAnimationOverlayCalculator
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
					return 0, ErrIntOverflowGlAnimationOverlayCalculator
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
					return 0, ErrIntOverflowGlAnimationOverlayCalculator
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
				return 0, ErrInvalidLengthGlAnimationOverlayCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupGlAnimationOverlayCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthGlAnimationOverlayCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthGlAnimationOverlayCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowGlAnimationOverlayCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupGlAnimationOverlayCalculator = fmt.Errorf("proto: unexpected end of group")
)
