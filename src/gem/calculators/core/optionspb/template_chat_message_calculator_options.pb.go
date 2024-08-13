// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/gem/calculators/core/optionspb/template_chat_message_calculator_options.proto

package optionspb

import (
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

type TemplateChatMessageCalculatorOptions struct {
	MessageTemplate     string `protobuf:"bytes,1,opt,name=message_template,json=messageTemplate,proto3" json:"message_template,omitempty"`
	PresetSystemPrompt  string `protobuf:"bytes,2,opt,name=preset_system_prompt,json=presetSystemPrompt,proto3" json:"preset_system_prompt,omitempty"`
	AddGenerationPrompt bool   `protobuf:"varint,3,opt,name=add_generation_prompt,json=addGenerationPrompt,proto3" json:"add_generation_prompt,omitempty"`
}

func (m *TemplateChatMessageCalculatorOptions) Reset()      { *m = TemplateChatMessageCalculatorOptions{} }
func (*TemplateChatMessageCalculatorOptions) ProtoMessage() {}
func (*TemplateChatMessageCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_39ea8a1bed8b27a2, []int{0}
}
func (m *TemplateChatMessageCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TemplateChatMessageCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TemplateChatMessageCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TemplateChatMessageCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TemplateChatMessageCalculatorOptions.Merge(m, src)
}
func (m *TemplateChatMessageCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *TemplateChatMessageCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_TemplateChatMessageCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_TemplateChatMessageCalculatorOptions proto.InternalMessageInfo

func (m *TemplateChatMessageCalculatorOptions) GetMessageTemplate() string {
	if m != nil {
		return m.MessageTemplate
	}
	return ""
}

func (m *TemplateChatMessageCalculatorOptions) GetPresetSystemPrompt() string {
	if m != nil {
		return m.PresetSystemPrompt
	}
	return ""
}

func (m *TemplateChatMessageCalculatorOptions) GetAddGenerationPrompt() bool {
	if m != nil {
		return m.AddGenerationPrompt
	}
	return false
}

func init() {
	proto.RegisterType((*TemplateChatMessageCalculatorOptions)(nil), "gml.gem.calculators.core.optionspb.TemplateChatMessageCalculatorOptions")
}

func init() {
	proto.RegisterFile("src/gem/calculators/core/optionspb/template_chat_message_calculator_options.proto", fileDescriptor_39ea8a1bed8b27a2)
}

var fileDescriptor_39ea8a1bed8b27a2 = []byte{
	// 302 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x90, 0xbf, 0x4a, 0x43, 0x31,
	0x14, 0xc6, 0x6f, 0x14, 0x44, 0xef, 0xa2, 0x5c, 0x15, 0x3a, 0x1d, 0x4a, 0x71, 0xa8, 0x4b, 0x22,
	0x3a, 0x3a, 0x69, 0x07, 0x27, 0x51, 0xab, 0x93, 0x4b, 0x48, 0xef, 0x3d, 0xa4, 0x85, 0xa4, 0x09,
	0x49, 0x44, 0xdc, 0x7c, 0x04, 0x1f, 0xc3, 0x37, 0xf0, 0x15, 0x1c, 0x3b, 0x76, 0xb4, 0xe9, 0xe2,
	0xd8, 0x47, 0x90, 0xde, 0x7f, 0xba, 0xb9, 0x25, 0xf9, 0xbe, 0xdf, 0xef, 0x84, 0x93, 0xde, 0x79,
	0x97, 0x33, 0x89, 0x9a, 0xe5, 0x42, 0xe5, 0x4f, 0x4a, 0x04, 0xe3, 0x3c, 0xcb, 0x8d, 0x43, 0x66,
	0x6c, 0x98, 0x98, 0xa9, 0xb7, 0x23, 0x16, 0x50, 0x5b, 0x25, 0x02, 0xf2, 0x7c, 0x2c, 0x02, 0xd7,
	0xe8, 0xbd, 0x90, 0xc8, 0x7f, 0x01, 0x5e, 0x57, 0xa9, 0x75, 0x26, 0x98, 0xac, 0x27, 0xb5, 0xa2,
	0x12, 0x35, 0xfd, 0xa3, 0xa4, 0x6b, 0x25, 0x6d, 0x95, 0xbd, 0x0f, 0x92, 0x1e, 0x3d, 0xd4, 0xda,
	0xc1, 0x58, 0x84, 0xeb, 0x4a, 0x3a, 0x68, 0x89, 0x9b, 0xaa, 0x9a, 0x1d, 0xa7, 0x7b, 0xcd, 0xc0,
	0xe6, 0x1b, 0x1d, 0xd2, 0x25, 0xfd, 0x9d, 0xe1, 0x6e, 0xfd, 0xde, 0x68, 0xb2, 0x93, 0xf4, 0xc0,
	0x3a, 0xf4, 0x18, 0xb8, 0x7f, 0xf1, 0x01, 0x35, 0xb7, 0xce, 0x68, 0x1b, 0x3a, 0x1b, 0x65, 0x3d,
	0xab, 0xb2, 0xfb, 0x32, 0xba, 0x2d, 0x93, 0xec, 0x34, 0x3d, 0x14, 0x45, 0xc1, 0x25, 0x4e, 0xd1,
	0x89, 0xf5, 0xbc, 0x06, 0xd9, 0xec, 0x92, 0xfe, 0xf6, 0x70, 0x5f, 0x14, 0xc5, 0x55, 0x9b, 0x55,
	0xcc, 0xe5, 0xf3, 0x6c, 0x01, 0xc9, 0x7c, 0x01, 0xc9, 0x6a, 0x01, 0xe4, 0x35, 0x02, 0x79, 0x8f,
	0x40, 0x3e, 0x23, 0x90, 0x59, 0x04, 0xf2, 0x15, 0x81, 0x7c, 0x47, 0x48, 0x56, 0x11, 0xc8, 0xdb,
	0x12, 0x92, 0xd9, 0x12, 0x92, 0xf9, 0x12, 0x92, 0xc7, 0x0b, 0x39, 0xd1, 0x0a, 0x83, 0x12, 0x23,
	0x4f, 0xc5, 0x84, 0x55, 0x37, 0xf6, 0xff, 0xee, 0xcf, 0xdb, 0xd3, 0x68, 0xab, 0xdc, 0xee, 0xd9,
	0x4f, 0x00, 0x00, 0x00, 0xff, 0xff, 0xb7, 0x65, 0x5a, 0x41, 0xb2, 0x01, 0x00, 0x00,
}

func (this *TemplateChatMessageCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*TemplateChatMessageCalculatorOptions)
	if !ok {
		that2, ok := that.(TemplateChatMessageCalculatorOptions)
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
	if this.MessageTemplate != that1.MessageTemplate {
		return false
	}
	if this.PresetSystemPrompt != that1.PresetSystemPrompt {
		return false
	}
	if this.AddGenerationPrompt != that1.AddGenerationPrompt {
		return false
	}
	return true
}
func (this *TemplateChatMessageCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&optionspb.TemplateChatMessageCalculatorOptions{")
	s = append(s, "MessageTemplate: "+fmt.Sprintf("%#v", this.MessageTemplate)+",\n")
	s = append(s, "PresetSystemPrompt: "+fmt.Sprintf("%#v", this.PresetSystemPrompt)+",\n")
	s = append(s, "AddGenerationPrompt: "+fmt.Sprintf("%#v", this.AddGenerationPrompt)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringTemplateChatMessageCalculatorOptions(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *TemplateChatMessageCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TemplateChatMessageCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *TemplateChatMessageCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.AddGenerationPrompt {
		i--
		if m.AddGenerationPrompt {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x18
	}
	if len(m.PresetSystemPrompt) > 0 {
		i -= len(m.PresetSystemPrompt)
		copy(dAtA[i:], m.PresetSystemPrompt)
		i = encodeVarintTemplateChatMessageCalculatorOptions(dAtA, i, uint64(len(m.PresetSystemPrompt)))
		i--
		dAtA[i] = 0x12
	}
	if len(m.MessageTemplate) > 0 {
		i -= len(m.MessageTemplate)
		copy(dAtA[i:], m.MessageTemplate)
		i = encodeVarintTemplateChatMessageCalculatorOptions(dAtA, i, uint64(len(m.MessageTemplate)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintTemplateChatMessageCalculatorOptions(dAtA []byte, offset int, v uint64) int {
	offset -= sovTemplateChatMessageCalculatorOptions(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *TemplateChatMessageCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.MessageTemplate)
	if l > 0 {
		n += 1 + l + sovTemplateChatMessageCalculatorOptions(uint64(l))
	}
	l = len(m.PresetSystemPrompt)
	if l > 0 {
		n += 1 + l + sovTemplateChatMessageCalculatorOptions(uint64(l))
	}
	if m.AddGenerationPrompt {
		n += 2
	}
	return n
}

func sovTemplateChatMessageCalculatorOptions(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozTemplateChatMessageCalculatorOptions(x uint64) (n int) {
	return sovTemplateChatMessageCalculatorOptions(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *TemplateChatMessageCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&TemplateChatMessageCalculatorOptions{`,
		`MessageTemplate:` + fmt.Sprintf("%v", this.MessageTemplate) + `,`,
		`PresetSystemPrompt:` + fmt.Sprintf("%v", this.PresetSystemPrompt) + `,`,
		`AddGenerationPrompt:` + fmt.Sprintf("%v", this.AddGenerationPrompt) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringTemplateChatMessageCalculatorOptions(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *TemplateChatMessageCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTemplateChatMessageCalculatorOptions
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
			return fmt.Errorf("proto: TemplateChatMessageCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TemplateChatMessageCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field MessageTemplate", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTemplateChatMessageCalculatorOptions
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthTemplateChatMessageCalculatorOptions
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthTemplateChatMessageCalculatorOptions
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.MessageTemplate = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field PresetSystemPrompt", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTemplateChatMessageCalculatorOptions
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				stringLen |= uint64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
			intStringLen := int(stringLen)
			if intStringLen < 0 {
				return ErrInvalidLengthTemplateChatMessageCalculatorOptions
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthTemplateChatMessageCalculatorOptions
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.PresetSystemPrompt = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field AddGenerationPrompt", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTemplateChatMessageCalculatorOptions
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
			m.AddGenerationPrompt = bool(v != 0)
		default:
			iNdEx = preIndex
			skippy, err := skipTemplateChatMessageCalculatorOptions(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTemplateChatMessageCalculatorOptions
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
func skipTemplateChatMessageCalculatorOptions(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTemplateChatMessageCalculatorOptions
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
					return 0, ErrIntOverflowTemplateChatMessageCalculatorOptions
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
					return 0, ErrIntOverflowTemplateChatMessageCalculatorOptions
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
				return 0, ErrInvalidLengthTemplateChatMessageCalculatorOptions
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupTemplateChatMessageCalculatorOptions
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthTemplateChatMessageCalculatorOptions
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthTemplateChatMessageCalculatorOptions        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTemplateChatMessageCalculatorOptions          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupTemplateChatMessageCalculatorOptions = fmt.Errorf("proto: unexpected end of group")
)