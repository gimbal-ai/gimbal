// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/image/feature_detector_calculator.proto

package image

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

type FeatureDetectorCalculatorOptions struct {
	OutputPatch  bool     `protobuf:"varint,1,opt,name=output_patch,json=outputPatch" json:"output_patch"`
	MaxFeatures  *int32   `protobuf:"varint,2,opt,name=max_features,json=maxFeatures,def=200" json:"max_features,omitempty"`
	PyramidLevel *int32   `protobuf:"varint,3,opt,name=pyramid_level,json=pyramidLevel,def=4" json:"pyramid_level,omitempty"`
	ScaleFactor  *float32 `protobuf:"fixed32,4,opt,name=scale_factor,json=scaleFactor,def=1.2" json:"scale_factor,omitempty"`
}

func (m *FeatureDetectorCalculatorOptions) Reset()      { *m = FeatureDetectorCalculatorOptions{} }
func (*FeatureDetectorCalculatorOptions) ProtoMessage() {}
func (*FeatureDetectorCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_d79d2d2e7a595757, []int{0}
}
func (m *FeatureDetectorCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *FeatureDetectorCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_FeatureDetectorCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *FeatureDetectorCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_FeatureDetectorCalculatorOptions.Merge(m, src)
}
func (m *FeatureDetectorCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *FeatureDetectorCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_FeatureDetectorCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_FeatureDetectorCalculatorOptions proto.InternalMessageInfo

const Default_FeatureDetectorCalculatorOptions_MaxFeatures int32 = 200
const Default_FeatureDetectorCalculatorOptions_PyramidLevel int32 = 4
const Default_FeatureDetectorCalculatorOptions_ScaleFactor float32 = 1.2

func (m *FeatureDetectorCalculatorOptions) GetOutputPatch() bool {
	if m != nil {
		return m.OutputPatch
	}
	return false
}

func (m *FeatureDetectorCalculatorOptions) GetMaxFeatures() int32 {
	if m != nil && m.MaxFeatures != nil {
		return *m.MaxFeatures
	}
	return Default_FeatureDetectorCalculatorOptions_MaxFeatures
}

func (m *FeatureDetectorCalculatorOptions) GetPyramidLevel() int32 {
	if m != nil && m.PyramidLevel != nil {
		return *m.PyramidLevel
	}
	return Default_FeatureDetectorCalculatorOptions_PyramidLevel
}

func (m *FeatureDetectorCalculatorOptions) GetScaleFactor() float32 {
	if m != nil && m.ScaleFactor != nil {
		return *m.ScaleFactor
	}
	return Default_FeatureDetectorCalculatorOptions_ScaleFactor
}

var E_FeatureDetectorCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*FeatureDetectorCalculatorOptions)(nil),
	Field:         278741680,
	Name:          "mediapipe.FeatureDetectorCalculatorOptions.ext",
	Tag:           "bytes,278741680,opt,name=ext",
	Filename:      "mediapipe/calculators/image/feature_detector_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_FeatureDetectorCalculatorOptions_Ext)
	proto.RegisterType((*FeatureDetectorCalculatorOptions)(nil), "mediapipe.FeatureDetectorCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/image/feature_detector_calculator.proto", fileDescriptor_d79d2d2e7a595757)
}

var fileDescriptor_d79d2d2e7a595757 = []byte{
	// 358 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x91, 0x31, 0x4b, 0xf3, 0x40,
	0x1c, 0xc6, 0xef, 0x9a, 0xbe, 0xf0, 0xbe, 0x49, 0xdf, 0x25, 0x53, 0x10, 0x39, 0x83, 0x48, 0x2d,
	0x08, 0x49, 0x0d, 0x82, 0x50, 0x70, 0xa9, 0xd2, 0x49, 0x50, 0x32, 0xba, 0x1c, 0x67, 0x7a, 0x4d,
	0x83, 0x39, 0xef, 0xb8, 0x5c, 0xb4, 0x6e, 0x0e, 0xba, 0xfb, 0x21, 0x1c, 0x1c, 0xfd, 0x18, 0x1d,
	0x3b, 0x76, 0x12, 0x7b, 0x5d, 0x1c, 0xbb, 0xb8, 0x4b, 0xda, 0xd0, 0x8a, 0x82, 0xae, 0xcf, 0xf3,
	0xdc, 0xef, 0x9e, 0x87, 0xbf, 0x79, 0xc0, 0x68, 0x37, 0x21, 0x22, 0x11, 0xd4, 0x8f, 0x48, 0x1a,
	0xe5, 0x29, 0x51, 0x5c, 0x66, 0x7e, 0xc2, 0x48, 0x4c, 0xfd, 0x1e, 0x25, 0x2a, 0x97, 0x14, 0x77,
	0xa9, 0xa2, 0x91, 0xe2, 0x12, 0xaf, 0x22, 0x9e, 0x90, 0x5c, 0x71, 0xfb, 0xdf, 0xf2, 0xf9, 0xda,
	0xd6, 0x8a, 0xd4, 0x93, 0x84, 0xd1, 0x6b, 0x2e, 0x2f, 0xfc, 0xaf, 0x0f, 0x36, 0x1f, 0x2b, 0xa6,
	0xdb, 0x59, 0x60, 0x8f, 0x4a, 0xea, 0xe1, 0x32, 0x73, 0x22, 0x54, 0xc2, 0x2f, 0x33, 0x7b, 0xdb,
	0xac, 0xf1, 0x5c, 0x89, 0x5c, 0x61, 0x41, 0x54, 0xd4, 0x77, 0xa0, 0x0b, 0x1b, 0x7f, 0xdb, 0xd5,
	0xe1, 0xcb, 0x06, 0x08, 0xad, 0x85, 0x73, 0x5a, 0x18, 0x76, 0xdd, 0xac, 0x31, 0x32, 0xc0, 0x65,
	0xcf, 0xcc, 0xa9, 0xb8, 0xb0, 0xf1, 0xa7, 0x65, 0x04, 0xcd, 0x66, 0x68, 0x31, 0x32, 0x28, 0x3f,
	0xca, 0xec, 0xba, 0xf9, 0x5f, 0xdc, 0x48, 0xc2, 0x92, 0x2e, 0x4e, 0xe9, 0x15, 0x4d, 0x1d, 0x63,
	0x1e, 0x84, 0x7b, 0x61, 0xad, 0xd4, 0x8f, 0x0b, 0xb9, 0xe0, 0x65, 0x11, 0x49, 0x29, 0xee, 0x91,
	0xa2, 0x99, 0x53, 0x75, 0x61, 0xa3, 0xd2, 0x32, 0x76, 0xbd, 0x20, 0xb4, 0xe6, 0x46, 0x67, 0xae,
	0x07, 0xd8, 0x34, 0xe8, 0x40, 0xd9, 0xeb, 0xde, 0x72, 0xb3, 0xf7, 0x6d, 0x85, 0xf3, 0x7c, 0xff,
	0x7e, 0x57, 0x34, 0xb7, 0x82, 0x9d, 0x4f, 0xb9, 0xdf, 0xc6, 0x87, 0x05, 0xb9, 0xcd, 0x46, 0x13,
	0x04, 0xc6, 0x13, 0x04, 0x66, 0x13, 0x04, 0x6f, 0x35, 0x82, 0x4f, 0x1a, 0xc1, 0xa1, 0x46, 0x70,
	0xa4, 0x11, 0x7c, 0xd5, 0x08, 0xbe, 0x69, 0x04, 0x66, 0x1a, 0xc1, 0x87, 0x29, 0x02, 0xa3, 0x29,
	0x02, 0xe3, 0x29, 0x02, 0x67, 0xfb, 0x71, 0xa2, 0xfa, 0xf9, 0xb9, 0x17, 0x71, 0xe6, 0xc7, 0x9c,
	0xc7, 0x29, 0xf5, 0x57, 0x47, 0xf9, 0xe1, 0xd0, 0x1f, 0x01, 0x00, 0x00, 0xff, 0xff, 0x3b, 0x42,
	0xf6, 0xb1, 0x06, 0x02, 0x00, 0x00,
}

func (this *FeatureDetectorCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*FeatureDetectorCalculatorOptions)
	if !ok {
		that2, ok := that.(FeatureDetectorCalculatorOptions)
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
	if this.OutputPatch != that1.OutputPatch {
		return false
	}
	if this.MaxFeatures != nil && that1.MaxFeatures != nil {
		if *this.MaxFeatures != *that1.MaxFeatures {
			return false
		}
	} else if this.MaxFeatures != nil {
		return false
	} else if that1.MaxFeatures != nil {
		return false
	}
	if this.PyramidLevel != nil && that1.PyramidLevel != nil {
		if *this.PyramidLevel != *that1.PyramidLevel {
			return false
		}
	} else if this.PyramidLevel != nil {
		return false
	} else if that1.PyramidLevel != nil {
		return false
	}
	if this.ScaleFactor != nil && that1.ScaleFactor != nil {
		if *this.ScaleFactor != *that1.ScaleFactor {
			return false
		}
	} else if this.ScaleFactor != nil {
		return false
	} else if that1.ScaleFactor != nil {
		return false
	}
	return true
}
func (this *FeatureDetectorCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 8)
	s = append(s, "&image.FeatureDetectorCalculatorOptions{")
	s = append(s, "OutputPatch: "+fmt.Sprintf("%#v", this.OutputPatch)+",\n")
	if this.MaxFeatures != nil {
		s = append(s, "MaxFeatures: "+valueToGoStringFeatureDetectorCalculator(this.MaxFeatures, "int32")+",\n")
	}
	if this.PyramidLevel != nil {
		s = append(s, "PyramidLevel: "+valueToGoStringFeatureDetectorCalculator(this.PyramidLevel, "int32")+",\n")
	}
	if this.ScaleFactor != nil {
		s = append(s, "ScaleFactor: "+valueToGoStringFeatureDetectorCalculator(this.ScaleFactor, "float32")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringFeatureDetectorCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *FeatureDetectorCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *FeatureDetectorCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *FeatureDetectorCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.ScaleFactor != nil {
		i -= 4
		encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(*m.ScaleFactor))))
		i--
		dAtA[i] = 0x25
	}
	if m.PyramidLevel != nil {
		i = encodeVarintFeatureDetectorCalculator(dAtA, i, uint64(*m.PyramidLevel))
		i--
		dAtA[i] = 0x18
	}
	if m.MaxFeatures != nil {
		i = encodeVarintFeatureDetectorCalculator(dAtA, i, uint64(*m.MaxFeatures))
		i--
		dAtA[i] = 0x10
	}
	i--
	if m.OutputPatch {
		dAtA[i] = 1
	} else {
		dAtA[i] = 0
	}
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func encodeVarintFeatureDetectorCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovFeatureDetectorCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *FeatureDetectorCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 2
	if m.MaxFeatures != nil {
		n += 1 + sovFeatureDetectorCalculator(uint64(*m.MaxFeatures))
	}
	if m.PyramidLevel != nil {
		n += 1 + sovFeatureDetectorCalculator(uint64(*m.PyramidLevel))
	}
	if m.ScaleFactor != nil {
		n += 5
	}
	return n
}

func sovFeatureDetectorCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozFeatureDetectorCalculator(x uint64) (n int) {
	return sovFeatureDetectorCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *FeatureDetectorCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&FeatureDetectorCalculatorOptions{`,
		`OutputPatch:` + fmt.Sprintf("%v", this.OutputPatch) + `,`,
		`MaxFeatures:` + valueToStringFeatureDetectorCalculator(this.MaxFeatures) + `,`,
		`PyramidLevel:` + valueToStringFeatureDetectorCalculator(this.PyramidLevel) + `,`,
		`ScaleFactor:` + valueToStringFeatureDetectorCalculator(this.ScaleFactor) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringFeatureDetectorCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *FeatureDetectorCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowFeatureDetectorCalculator
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
			return fmt.Errorf("proto: FeatureDetectorCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: FeatureDetectorCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field OutputPatch", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowFeatureDetectorCalculator
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
			m.OutputPatch = bool(v != 0)
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field MaxFeatures", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowFeatureDetectorCalculator
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
			m.MaxFeatures = &v
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field PyramidLevel", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowFeatureDetectorCalculator
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
			m.PyramidLevel = &v
		case 4:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field ScaleFactor", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			v2 := float32(math.Float32frombits(v))
			m.ScaleFactor = &v2
		default:
			iNdEx = preIndex
			skippy, err := skipFeatureDetectorCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthFeatureDetectorCalculator
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
func skipFeatureDetectorCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowFeatureDetectorCalculator
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
					return 0, ErrIntOverflowFeatureDetectorCalculator
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
					return 0, ErrIntOverflowFeatureDetectorCalculator
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
				return 0, ErrInvalidLengthFeatureDetectorCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupFeatureDetectorCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthFeatureDetectorCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthFeatureDetectorCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowFeatureDetectorCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupFeatureDetectorCalculator = fmt.Errorf("proto: unexpected end of group")
)
