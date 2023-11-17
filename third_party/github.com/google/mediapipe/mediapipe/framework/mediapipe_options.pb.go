// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/mediapipe_options.proto

package framework

import (
	fmt "fmt"
	github_com_gogo_protobuf_proto "github.com/gogo/protobuf/proto"
	proto "github.com/gogo/protobuf/proto"
	io "io"
	math "math"
	math_bits "math/bits"
	reflect "reflect"
	sort "sort"
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

type MediaPipeOptions struct {
	proto.XXX_InternalExtensions `json:"-"`
}

func (m *MediaPipeOptions) Reset()      { *m = MediaPipeOptions{} }
func (*MediaPipeOptions) ProtoMessage() {}
func (*MediaPipeOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_645434bc481a450e, []int{0}
}

var extRange_MediaPipeOptions = []proto.ExtensionRange{
	{Start: 20000, End: 536870911},
}

func (*MediaPipeOptions) ExtensionRangeArray() []proto.ExtensionRange {
	return extRange_MediaPipeOptions
}

func (m *MediaPipeOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *MediaPipeOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_MediaPipeOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *MediaPipeOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_MediaPipeOptions.Merge(m, src)
}
func (m *MediaPipeOptions) XXX_Size() int {
	return m.Size()
}
func (m *MediaPipeOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_MediaPipeOptions.DiscardUnknown(m)
}

var xxx_messageInfo_MediaPipeOptions proto.InternalMessageInfo

func init() {
	proto.RegisterType((*MediaPipeOptions)(nil), "mediapipe.MediaPipeOptions")
}

func init() {
	proto.RegisterFile("mediapipe/framework/mediapipe_options.proto", fileDescriptor_645434bc481a450e)
}

var fileDescriptor_645434bc481a450e = []byte{
	// 195 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xd2, 0xce, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x2b, 0x4a, 0xcc, 0x4d, 0x2d, 0xcf, 0x2f, 0xca, 0xd6,
	0x87, 0x8b, 0xc5, 0xe7, 0x17, 0x94, 0x64, 0xe6, 0xe7, 0x15, 0xeb, 0x15, 0x14, 0xe5, 0x97, 0xe4,
	0x0b, 0x71, 0xc2, 0x25, 0x94, 0xe4, 0xb8, 0x04, 0x7c, 0x41, 0x9c, 0x80, 0xcc, 0x82, 0x54, 0x7f,
	0x88, 0x22, 0x2d, 0x2e, 0x8e, 0x05, 0x73, 0x18, 0x05, 0x1a, 0x1a, 0x1a, 0x1a, 0x98, 0x9c, 0x66,
	0x30, 0x5e, 0x78, 0x28, 0xc7, 0x70, 0xe3, 0xa1, 0x1c, 0xc3, 0x87, 0x87, 0x72, 0x8c, 0x0d, 0x8f,
	0xe4, 0x18, 0x57, 0x3c, 0x92, 0x63, 0x3c, 0xf1, 0x48, 0x8e, 0xf1, 0xc2, 0x23, 0x39, 0xc6, 0x07,
	0x8f, 0xe4, 0x18, 0x5f, 0x3c, 0x92, 0x63, 0xf8, 0xf0, 0x48, 0x8e, 0x71, 0xc2, 0x63, 0x39, 0x86,
	0x0b, 0x8f, 0xe5, 0x18, 0x6e, 0x3c, 0x96, 0x63, 0xe0, 0x92, 0x4a, 0xce, 0xcf, 0xd5, 0x4b, 0xcf,
	0xcf, 0x4f, 0xcf, 0x49, 0xd5, 0x83, 0x5b, 0x06, 0xb1, 0xdd, 0x49, 0x14, 0xdd, 0xc2, 0x00, 0x90,
	0x70, 0x94, 0x7e, 0x7a, 0x66, 0x49, 0x46, 0x69, 0x92, 0x5e, 0x72, 0x7e, 0xae, 0x3e, 0x44, 0x27,
	0xc2, 0xfd, 0xfa, 0x58, 0x7c, 0x07, 0x08, 0x00, 0x00, 0xff, 0xff, 0xe1, 0x6e, 0x13, 0xfe, 0xf3,
	0x00, 0x00, 0x00,
}

func (this *MediaPipeOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*MediaPipeOptions)
	if !ok {
		that2, ok := that.(MediaPipeOptions)
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
	thismap := github_com_gogo_protobuf_proto.GetUnsafeExtensionsMap(this)
	thatmap := github_com_gogo_protobuf_proto.GetUnsafeExtensionsMap(that1)
	for k, v := range thismap {
		if v2, ok := thatmap[k]; ok {
			if !v.Equal(&v2) {
				return false
			}
		} else {
			return false
		}
	}
	for k, _ := range thatmap {
		if _, ok := thismap[k]; !ok {
			return false
		}
	}
	return true
}
func (this *MediaPipeOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 4)
	s = append(s, "&framework.MediaPipeOptions{")
	s = append(s, "XXX_InternalExtensions: "+extensionToGoStringMediapipeOptions(this)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringMediapipeOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func extensionToGoStringMediapipeOptions(m github_com_gogo_protobuf_proto.Message) string {
	e := github_com_gogo_protobuf_proto.GetUnsafeExtensionsMap(m)
	if e == nil {
		return "nil"
	}
	s := "proto.NewUnsafeXXX_InternalExtensions(map[int32]proto.Extension{"
	keys := make([]int, 0, len(e))
	for k := range e {
		keys = append(keys, int(k))
	}
	sort.Ints(keys)
	ss := []string{}
	for _, k := range keys {
		ss = append(ss, strconv.Itoa(k)+": "+e[int32(k)].GoString())
	}
	s += strings.Join(ss, ",") + "})"
	return s
}
func (m *MediaPipeOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *MediaPipeOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *MediaPipeOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if n, err := github_com_gogo_protobuf_proto.EncodeInternalExtensionBackwards(m, dAtA[:i]); err != nil {
		return 0, err
	} else {
		i -= n
	}
	return len(dAtA) - i, nil
}

func encodeVarintMediapipeOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovMediapipeOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *MediaPipeOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += github_com_gogo_protobuf_proto.SizeOfInternalExtension(m)
	return n
}

func sovMediapipeOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozMediapipeOptions(x uint64) (n int) {
	return sovMediapipeOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *MediaPipeOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&MediaPipeOptions{`,
		`XXX_InternalExtensions:` + github_com_gogo_protobuf_proto.StringFromInternalExtension(this) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringMediapipeOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *MediaPipeOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMediapipeOptions
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
			return fmt.Errorf("proto: MediaPipeOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: MediaPipeOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		default:
			if (fieldNum >= 20000) && (fieldNum < 536870912) {
				var sizeOfWire int
				for {
					sizeOfWire++
					wire >>= 7
					if wire == 0 {
						break
					}
				}
				iNdEx -= sizeOfWire
				skippy, err := skipMediapipeOptions(dAtA[iNdEx:])
				if err != nil {
					return err
				}
				if (skippy < 0) || (iNdEx+skippy) < 0 {
					return ErrInvalidLengthMediapipeOptions
				}
				if (iNdEx + skippy) > l {
					return io.ErrUnexpectedEOF
				}
				github_com_gogo_protobuf_proto.AppendExtension(m, int32(fieldNum), dAtA[iNdEx:iNdEx+skippy])
				iNdEx += skippy
			} else {
				iNdEx = preIndex
				skippy, err := skipMediapipeOptions(dAtA[iNdEx:])
				if err != nil {
					return err
				}
				if (skippy < 0) || (iNdEx+skippy) < 0 {
					return ErrInvalidLengthMediapipeOptions
				}
				if (iNdEx + skippy) > l {
					return io.ErrUnexpectedEOF
				}
				iNdEx += skippy
			}
		}
	}

	if iNdEx > l {
		return io.ErrUnexpectedEOF
	}
	return nil
}
func skipMediapipeOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowMediapipeOptions
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
					return 0, ErrIntOverflowMediapipeOptions
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
					return 0, ErrIntOverflowMediapipeOptions
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
				return 0, ErrInvalidLengthMediapipeOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupMediapipeOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthMediapipeOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthMediapipeOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowMediapipeOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupMediapipeOptions = fmt.Errorf("proto: unexpected end of group")
)