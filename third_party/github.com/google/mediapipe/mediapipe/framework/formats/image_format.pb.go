// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/formats/image_format.proto

package formats

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
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

type ImageFormat_Format int32

const (
	FORMAT_UNKNOWN     ImageFormat_Format = 0
	FORMAT_SRGB        ImageFormat_Format = 1
	FORMAT_SRGBA       ImageFormat_Format = 2
	FORMAT_GRAY8       ImageFormat_Format = 3
	FORMAT_GRAY16      ImageFormat_Format = 4
	FORMAT_YCBCR420P   ImageFormat_Format = 5
	FORMAT_YCBCR420P10 ImageFormat_Format = 6
	FORMAT_SRGB48      ImageFormat_Format = 7
	FORMAT_SRGBA64     ImageFormat_Format = 8
	FORMAT_VEC32F1     ImageFormat_Format = 9
	FORMAT_VEC32F2     ImageFormat_Format = 12
	FORMAT_VEC32F4     ImageFormat_Format = 13
	FORMAT_LAB8        ImageFormat_Format = 10
	FORMAT_SBGRA       ImageFormat_Format = 11
)

var ImageFormat_Format_name = map[int32]string{
	0:  "FORMAT_UNKNOWN",
	1:  "FORMAT_SRGB",
	2:  "FORMAT_SRGBA",
	3:  "FORMAT_GRAY8",
	4:  "FORMAT_GRAY16",
	5:  "FORMAT_YCBCR420P",
	6:  "FORMAT_YCBCR420P10",
	7:  "FORMAT_SRGB48",
	8:  "FORMAT_SRGBA64",
	9:  "FORMAT_VEC32F1",
	12: "FORMAT_VEC32F2",
	13: "FORMAT_VEC32F4",
	10: "FORMAT_LAB8",
	11: "FORMAT_SBGRA",
}

var ImageFormat_Format_value = map[string]int32{
	"FORMAT_UNKNOWN":     0,
	"FORMAT_SRGB":        1,
	"FORMAT_SRGBA":       2,
	"FORMAT_GRAY8":       3,
	"FORMAT_GRAY16":      4,
	"FORMAT_YCBCR420P":   5,
	"FORMAT_YCBCR420P10": 6,
	"FORMAT_SRGB48":      7,
	"FORMAT_SRGBA64":     8,
	"FORMAT_VEC32F1":     9,
	"FORMAT_VEC32F2":     12,
	"FORMAT_VEC32F4":     13,
	"FORMAT_LAB8":        10,
	"FORMAT_SBGRA":       11,
}

func (x ImageFormat_Format) Enum() *ImageFormat_Format {
	p := new(ImageFormat_Format)
	*p = x
	return p
}

func (x ImageFormat_Format) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(ImageFormat_Format_name, int32(x))
}

func (x *ImageFormat_Format) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(ImageFormat_Format_value, data, "ImageFormat_Format")
	if err != nil {
		return err
	}
	*x = ImageFormat_Format(value)
	return nil
}

func (ImageFormat_Format) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_8b7f1e73bffbf5b2, []int{0, 0}
}

type ImageFormat struct {
}

func (m *ImageFormat) Reset()      { *m = ImageFormat{} }
func (*ImageFormat) ProtoMessage() {}
func (*ImageFormat) Descriptor() ([]byte, []int) {
	return fileDescriptor_8b7f1e73bffbf5b2, []int{0}
}
func (m *ImageFormat) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ImageFormat) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ImageFormat.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ImageFormat) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ImageFormat.Merge(m, src)
}
func (m *ImageFormat) XXX_Size() int {
	return m.Size()
}
func (m *ImageFormat) XXX_DiscardUnknown() {
	xxx_messageInfo_ImageFormat.DiscardUnknown(m)
}

var xxx_messageInfo_ImageFormat proto.InternalMessageInfo

func init() {
	proto.RegisterEnum("mediapipe.ImageFormat_Format", ImageFormat_Format_name, ImageFormat_Format_value)
	proto.RegisterType((*ImageFormat)(nil), "mediapipe.ImageFormat")
}

func init() {
	proto.RegisterFile("mediapipe/framework/formats/image_format.proto", fileDescriptor_8b7f1e73bffbf5b2)
}

var fileDescriptor_8b7f1e73bffbf5b2 = []byte{
	// 334 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x91, 0xbd, 0x4e, 0x02, 0x41,
	0x14, 0x85, 0x67, 0x50, 0x51, 0x06, 0xd0, 0xeb, 0xc4, 0x58, 0xde, 0x82, 0x07, 0xd8, 0x05, 0xdc,
	0x20, 0xed, 0x2e, 0x11, 0x62, 0x54, 0x20, 0xeb, 0x5f, 0xb0, 0x21, 0x2b, 0x2e, 0xeb, 0x46, 0x27,
	0x43, 0x56, 0x8c, 0xad, 0x8f, 0x60, 0x69, 0x6f, 0xc3, 0xa3, 0x58, 0x52, 0x52, 0xca, 0xd0, 0x58,
	0xf2, 0x08, 0x06, 0x96, 0xc8, 0x44, 0x12, 0xbb, 0x99, 0x2f, 0x67, 0xe6, 0xde, 0x93, 0x8f, 0x19,
	0xc2, 0xbf, 0x0b, 0xbd, 0x5e, 0xd8, 0xf3, 0xcd, 0x6e, 0xe4, 0x09, 0xff, 0x45, 0x46, 0x0f, 0x66,
	0x57, 0x46, 0xc2, 0xeb, 0x3f, 0x99, 0xa1, 0xf0, 0x02, 0xbf, 0x1d, 0xdf, 0x8c, 0x5e, 0x24, 0xfb,
	0x92, 0xa7, 0x7e, 0xf3, 0xb9, 0x41, 0x82, 0xa5, 0x8f, 0x67, 0x89, 0xea, 0x3c, 0x90, 0x7b, 0x4f,
	0xb0, 0x64, 0x7c, 0xe4, 0x9c, 0x6d, 0x57, 0x1b, 0xee, 0x99, 0x7d, 0xd1, 0xbe, 0xac, 0x9f, 0xd4,
	0x1b, 0xd7, 0x75, 0x20, 0x7c, 0x87, 0xa5, 0x17, 0xec, 0xdc, 0xad, 0x39, 0x40, 0x39, 0xb0, 0x8c,
	0x06, 0x6c, 0x48, 0x68, 0xa4, 0xe6, 0xda, 0xad, 0x32, 0xac, 0xf1, 0x5d, 0x96, 0xd5, 0x48, 0xa1,
	0x04, 0xeb, 0x7c, 0x8f, 0xc1, 0x02, 0xb5, 0x2a, 0x4e, 0xc5, 0xb5, 0x8a, 0xf9, 0x26, 0x6c, 0xf0,
	0x7d, 0xc6, 0xff, 0xd2, 0x42, 0x1e, 0x92, 0xda, 0x07, 0xb3, 0x21, 0x56, 0x19, 0x36, 0xb5, 0xe5,
	0xe6, 0x73, 0x4b, 0x16, 0x6c, 0x69, 0xec, 0xea, 0xa8, 0x72, 0x50, 0xac, 0x16, 0x20, 0xb5, 0xc2,
	0x8a, 0x90, 0x59, 0x61, 0x16, 0x64, 0xb5, 0x62, 0xa7, 0xb6, 0x53, 0x06, 0xa6, 0x17, 0x73, 0x6a,
	0xae, 0x0d, 0x69, 0xe7, 0x83, 0x0e, 0xc7, 0x48, 0x46, 0x63, 0x24, 0xd3, 0x31, 0xd2, 0x57, 0x85,
	0x74, 0xa0, 0x90, 0x7e, 0x2a, 0xa4, 0x43, 0x85, 0xf4, 0x4b, 0x21, 0xfd, 0x56, 0x48, 0xa6, 0x0a,
	0xe9, 0xdb, 0x04, 0xc9, 0x70, 0x82, 0x64, 0x34, 0x41, 0xc2, 0x72, 0x1d, 0x29, 0x8c, 0x40, 0xca,
	0xe0, 0xd1, 0x5f, 0x7a, 0x32, 0x16, 0x76, 0x62, 0x21, 0x0e, 0x68, 0x0a, 0x9a, 0x33, 0x72, 0x73,
	0x18, 0x84, 0xfd, 0xfb, 0xe7, 0x5b, 0xa3, 0x23, 0x85, 0x19, 0x3f, 0x36, 0x97, 0x92, 0xff, 0xd1,
	0xfd, 0x13, 0x00, 0x00, 0xff, 0xff, 0x5e, 0xd1, 0xee, 0xba, 0x0c, 0x02, 0x00, 0x00,
}

func (x ImageFormat_Format) String() string {
	s, ok := ImageFormat_Format_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *ImageFormat) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ImageFormat)
	if !ok {
		that2, ok := that.(ImageFormat)
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
	return true
}
func (this *ImageFormat) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 4)
	s = append(s, "&formats.ImageFormat{")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringImageFormat(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *ImageFormat) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ImageFormat) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ImageFormat) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	return len(dAtA) - i, nil
}

func encodeVarintImageFormat(dAtA []byte, offset int, v uint64) int {
	offset -= sovImageFormat(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ImageFormat) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	return n
}

func sovImageFormat(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozImageFormat(x uint64) (n int) {
	return sovImageFormat(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ImageFormat) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ImageFormat{`,
		`}`,
	}, "")
	return s
}
func valueToStringImageFormat(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ImageFormat) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowImageFormat
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
			return fmt.Errorf("proto: ImageFormat: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ImageFormat: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		default:
			iNdEx = preIndex
			skippy, err := skipImageFormat(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthImageFormat
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
func skipImageFormat(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowImageFormat
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
					return 0, ErrIntOverflowImageFormat
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
					return 0, ErrIntOverflowImageFormat
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
				return 0, ErrInvalidLengthImageFormat
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupImageFormat
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthImageFormat
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthImageFormat        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowImageFormat          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupImageFormat = fmt.Errorf("proto: unexpected end of group")
)
