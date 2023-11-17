// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/util/landmarks_to_detection_calculator.proto

package util

import (
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

type LandmarksToDetectionCalculatorOptions struct {
	SelectedLandmarkIndices []int32 `protobuf:"varint,1,rep,name=selected_landmark_indices,json=selectedLandmarkIndices" json:"selected_landmark_indices,omitempty"`
}

func (m *LandmarksToDetectionCalculatorOptions) Reset()      { *m = LandmarksToDetectionCalculatorOptions{} }
func (*LandmarksToDetectionCalculatorOptions) ProtoMessage() {}
func (*LandmarksToDetectionCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_5de3e56bceb1944e, []int{0}
}
func (m *LandmarksToDetectionCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *LandmarksToDetectionCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_LandmarksToDetectionCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *LandmarksToDetectionCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_LandmarksToDetectionCalculatorOptions.Merge(m, src)
}
func (m *LandmarksToDetectionCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *LandmarksToDetectionCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_LandmarksToDetectionCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_LandmarksToDetectionCalculatorOptions proto.InternalMessageInfo

func (m *LandmarksToDetectionCalculatorOptions) GetSelectedLandmarkIndices() []int32 {
	if m != nil {
		return m.SelectedLandmarkIndices
	}
	return nil
}

var E_LandmarksToDetectionCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*LandmarksToDetectionCalculatorOptions)(nil),
	Field:         260199669,
	Name:          "mediapipe.LandmarksToDetectionCalculatorOptions.ext",
	Tag:           "bytes,260199669,opt,name=ext",
	Filename:      "mediapipe/calculators/util/landmarks_to_detection_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_LandmarksToDetectionCalculatorOptions_Ext)
	proto.RegisterType((*LandmarksToDetectionCalculatorOptions)(nil), "mediapipe.LandmarksToDetectionCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/util/landmarks_to_detection_calculator.proto", fileDescriptor_5de3e56bceb1944e)
}

var fileDescriptor_5de3e56bceb1944e = []byte{
	// 281 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0x72, 0xca, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0x2f, 0x2d, 0xc9, 0xcc, 0xd1, 0xcf, 0x49, 0xcc, 0x4b, 0xc9, 0x4d, 0x2c, 0xca, 0x2e,
	0x8e, 0x2f, 0xc9, 0x8f, 0x4f, 0x49, 0x2d, 0x49, 0x4d, 0x2e, 0xc9, 0xcc, 0xcf, 0x8b, 0x47, 0xa8,
	0xd3, 0x2b, 0x28, 0xca, 0x2f, 0xc9, 0x17, 0xe2, 0x84, 0x9b, 0x21, 0xa5, 0x82, 0x30, 0x2e, 0xad,
	0x28, 0x31, 0x37, 0xb5, 0x3c, 0xbf, 0x28, 0x5b, 0x1f, 0x5d, 0x83, 0xd2, 0x09, 0x46, 0x2e, 0x55,
	0x1f, 0x98, 0xe1, 0x21, 0xf9, 0x2e, 0x30, 0xa3, 0x9d, 0xe1, 0x0a, 0xfd, 0x0b, 0x40, 0xfc, 0x62,
	0x21, 0x2b, 0x2e, 0xc9, 0xe2, 0xd4, 0x9c, 0xd4, 0xe4, 0x92, 0xd4, 0x94, 0x78, 0x98, 0x73, 0xe2,
	0x33, 0xf3, 0x52, 0x32, 0x93, 0x53, 0x8b, 0x25, 0x18, 0x15, 0x98, 0x35, 0x58, 0x83, 0xc4, 0x61,
	0x0a, 0x60, 0x26, 0x7a, 0x42, 0xa4, 0x8d, 0x92, 0xb9, 0x98, 0x53, 0x2b, 0x4a, 0x84, 0x64, 0xf4,
	0xe0, 0x6e, 0xd2, 0xc3, 0xb0, 0x40, 0xe2, 0xeb, 0xca, 0xce, 0x1a, 0x05, 0x46, 0x0d, 0x6e, 0x23,
	0x03, 0x24, 0x65, 0x44, 0xb9, 0x2d, 0x08, 0x64, 0xba, 0x53, 0xce, 0x85, 0x87, 0x72, 0x0c, 0x37,
	0x1e, 0xca, 0x31, 0x7c, 0x78, 0x28, 0xc7, 0xd8, 0xf0, 0x48, 0x8e, 0x71, 0xc5, 0x23, 0x39, 0xc6,
	0x13, 0x8f, 0xe4, 0x18, 0x2f, 0x3c, 0x92, 0x63, 0x7c, 0xf0, 0x48, 0x8e, 0xf1, 0xc5, 0x23, 0x39,
	0x86, 0x0f, 0x8f, 0xe4, 0x18, 0x27, 0x3c, 0x96, 0x63, 0xb8, 0xf0, 0x58, 0x8e, 0xe1, 0xc6, 0x63,
	0x39, 0x86, 0x28, 0xb3, 0xf4, 0xcc, 0x92, 0x8c, 0xd2, 0x24, 0xbd, 0xe4, 0xfc, 0x5c, 0xfd, 0xf4,
	0xfc, 0xfc, 0xf4, 0x9c, 0x54, 0x7d, 0x44, 0xc0, 0xe1, 0x8e, 0x11, 0x40, 0x00, 0x00, 0x00, 0xff,
	0xff, 0xa6, 0x7f, 0x0b, 0x2b, 0xae, 0x01, 0x00, 0x00,
}

func (this *LandmarksToDetectionCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*LandmarksToDetectionCalculatorOptions)
	if !ok {
		that2, ok := that.(LandmarksToDetectionCalculatorOptions)
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
	if len(this.SelectedLandmarkIndices) != len(that1.SelectedLandmarkIndices) {
		return false
	}
	for i := range this.SelectedLandmarkIndices {
		if this.SelectedLandmarkIndices[i] != that1.SelectedLandmarkIndices[i] {
			return false
		}
	}
	return true
}
func (this *LandmarksToDetectionCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&util.LandmarksToDetectionCalculatorOptions{")
	if this.SelectedLandmarkIndices != nil {
		s = append(s, "SelectedLandmarkIndices: "+fmt.Sprintf("%#v", this.SelectedLandmarkIndices)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringLandmarksToDetectionCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *LandmarksToDetectionCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *LandmarksToDetectionCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *LandmarksToDetectionCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.SelectedLandmarkIndices) > 0 {
		for iNdEx := len(m.SelectedLandmarkIndices) - 1; iNdEx >= 0; iNdEx-- {
			i = encodeVarintLandmarksToDetectionCalculator(dAtA, i, uint64(m.SelectedLandmarkIndices[iNdEx]))
			i--
			dAtA[i] = 0x8
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintLandmarksToDetectionCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovLandmarksToDetectionCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *LandmarksToDetectionCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.SelectedLandmarkIndices) > 0 {
		for _, e := range m.SelectedLandmarkIndices {
			n += 1 + sovLandmarksToDetectionCalculator(uint64(e))
		}
	}
	return n
}

func sovLandmarksToDetectionCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozLandmarksToDetectionCalculator(x uint64) (n int) {
	return sovLandmarksToDetectionCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *LandmarksToDetectionCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&LandmarksToDetectionCalculatorOptions{`,
		`SelectedLandmarkIndices:` + fmt.Sprintf("%v", this.SelectedLandmarkIndices) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringLandmarksToDetectionCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *LandmarksToDetectionCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLandmarksToDetectionCalculator
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
			return fmt.Errorf("proto: LandmarksToDetectionCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: LandmarksToDetectionCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType == 0 {
				var v int32
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowLandmarksToDetectionCalculator
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
				m.SelectedLandmarkIndices = append(m.SelectedLandmarkIndices, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowLandmarksToDetectionCalculator
					}
					if iNdEx >= l {
						return io.ErrUnexpectedEOF
					}
					b := dAtA[iNdEx]
					iNdEx++
					packedLen |= int(b&0x7F) << shift
					if b < 0x80 {
						break
					}
				}
				if packedLen < 0 {
					return ErrInvalidLengthLandmarksToDetectionCalculator
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthLandmarksToDetectionCalculator
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				var count int
				for _, integer := range dAtA[iNdEx:postIndex] {
					if integer < 128 {
						count++
					}
				}
				elementCount = count
				if elementCount != 0 && len(m.SelectedLandmarkIndices) == 0 {
					m.SelectedLandmarkIndices = make([]int32, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v int32
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowLandmarksToDetectionCalculator
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
					m.SelectedLandmarkIndices = append(m.SelectedLandmarkIndices, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field SelectedLandmarkIndices", wireType)
			}
		default:
			iNdEx = preIndex
			skippy, err := skipLandmarksToDetectionCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthLandmarksToDetectionCalculator
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
func skipLandmarksToDetectionCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowLandmarksToDetectionCalculator
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
					return 0, ErrIntOverflowLandmarksToDetectionCalculator
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
					return 0, ErrIntOverflowLandmarksToDetectionCalculator
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
				return 0, ErrInvalidLengthLandmarksToDetectionCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupLandmarksToDetectionCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthLandmarksToDetectionCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthLandmarksToDetectionCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowLandmarksToDetectionCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupLandmarksToDetectionCalculator = fmt.Errorf("proto: unexpected end of group")
)