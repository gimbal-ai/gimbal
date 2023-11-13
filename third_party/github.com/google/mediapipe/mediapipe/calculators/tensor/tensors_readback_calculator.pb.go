// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/tensor/tensors_readback_calculator.proto

package tensor

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

type TensorsReadbackCalculatorOptions struct {
	TensorShape []*TensorsReadbackCalculatorOptions_TensorShape `protobuf:"bytes,1,rep,name=tensor_shape,json=tensorShape" json:"tensor_shape,omitempty"`
}

func (m *TensorsReadbackCalculatorOptions) Reset()      { *m = TensorsReadbackCalculatorOptions{} }
func (*TensorsReadbackCalculatorOptions) ProtoMessage() {}
func (*TensorsReadbackCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_883823a6e8ce6f5a, []int{0}
}
func (m *TensorsReadbackCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TensorsReadbackCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TensorsReadbackCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TensorsReadbackCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TensorsReadbackCalculatorOptions.Merge(m, src)
}
func (m *TensorsReadbackCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *TensorsReadbackCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_TensorsReadbackCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_TensorsReadbackCalculatorOptions proto.InternalMessageInfo

func (m *TensorsReadbackCalculatorOptions) GetTensorShape() []*TensorsReadbackCalculatorOptions_TensorShape {
	if m != nil {
		return m.TensorShape
	}
	return nil
}

var E_TensorsReadbackCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*TensorsReadbackCalculatorOptions)(nil),
	Field:         514750372,
	Name:          "mediapipe.TensorsReadbackCalculatorOptions.ext",
	Tag:           "bytes,514750372,opt,name=ext",
	Filename:      "mediapipe/calculators/tensor/tensors_readback_calculator.proto",
}

type TensorsReadbackCalculatorOptions_TensorShape struct {
	Dims []int32 `protobuf:"varint,1,rep,packed,name=dims" json:"dims,omitempty"`
}

func (m *TensorsReadbackCalculatorOptions_TensorShape) Reset() {
	*m = TensorsReadbackCalculatorOptions_TensorShape{}
}
func (*TensorsReadbackCalculatorOptions_TensorShape) ProtoMessage() {}
func (*TensorsReadbackCalculatorOptions_TensorShape) Descriptor() ([]byte, []int) {
	return fileDescriptor_883823a6e8ce6f5a, []int{0, 0}
}
func (m *TensorsReadbackCalculatorOptions_TensorShape) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *TensorsReadbackCalculatorOptions_TensorShape) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_TensorsReadbackCalculatorOptions_TensorShape.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *TensorsReadbackCalculatorOptions_TensorShape) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TensorsReadbackCalculatorOptions_TensorShape.Merge(m, src)
}
func (m *TensorsReadbackCalculatorOptions_TensorShape) XXX_Size() int {
	return m.Size()
}
func (m *TensorsReadbackCalculatorOptions_TensorShape) XXX_DiscardUnknown() {
	xxx_messageInfo_TensorsReadbackCalculatorOptions_TensorShape.DiscardUnknown(m)
}

var xxx_messageInfo_TensorsReadbackCalculatorOptions_TensorShape proto.InternalMessageInfo

func (m *TensorsReadbackCalculatorOptions_TensorShape) GetDims() []int32 {
	if m != nil {
		return m.Dims
	}
	return nil
}

func init() {
	proto.RegisterExtension(E_TensorsReadbackCalculatorOptions_Ext)
	proto.RegisterType((*TensorsReadbackCalculatorOptions)(nil), "mediapipe.TensorsReadbackCalculatorOptions")
	proto.RegisterType((*TensorsReadbackCalculatorOptions_TensorShape)(nil), "mediapipe.TensorsReadbackCalculatorOptions.TensorShape")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/tensor/tensors_readback_calculator.proto", fileDescriptor_883823a6e8ce6f5a)
}

var fileDescriptor_883823a6e8ce6f5a = []byte{
	// 290 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xe2, 0xb2, 0xcb, 0x4d, 0x4d, 0xc9,
	0x4c, 0x2c, 0xc8, 0x2c, 0x48, 0xd5, 0x4f, 0x4e, 0xcc, 0x49, 0x2e, 0xcd, 0x49, 0x2c, 0xc9, 0x2f,
	0x2a, 0xd6, 0x2f, 0x49, 0xcd, 0x2b, 0xce, 0x2f, 0x82, 0x52, 0xc5, 0xf1, 0x45, 0xa9, 0x89, 0x29,
	0x49, 0x89, 0xc9, 0xd9, 0xf1, 0x08, 0x35, 0x7a, 0x05, 0x45, 0xf9, 0x25, 0xf9, 0x42, 0x9c, 0x70,
	0xfd, 0x52, 0x2a, 0x08, 0xa3, 0xd2, 0x8a, 0x12, 0x73, 0x53, 0xcb, 0xf3, 0x8b, 0xb2, 0xf5, 0xd1,
	0x35, 0x28, 0xb5, 0x31, 0x71, 0x29, 0x84, 0x40, 0x8c, 0x0d, 0x82, 0x9a, 0xea, 0x0c, 0x57, 0xe3,
	0x5f, 0x50, 0x92, 0x99, 0x9f, 0x57, 0x2c, 0x14, 0xc5, 0xc5, 0x03, 0xb1, 0x3a, 0xbe, 0x38, 0x23,
	0xb1, 0x20, 0x55, 0x82, 0x51, 0x81, 0x59, 0x83, 0xdb, 0xc8, 0x5c, 0x0f, 0x6e, 0x83, 0x1e, 0x21,
	0x23, 0xa0, 0x0a, 0x82, 0x41, 0xda, 0x83, 0xb8, 0x4b, 0x10, 0x1c, 0x29, 0x55, 0x2e, 0x6e, 0x24,
	0x39, 0x21, 0x31, 0x2e, 0x96, 0x94, 0xcc, 0xdc, 0x62, 0xb0, 0x15, 0xac, 0x4e, 0x4c, 0x02, 0x8c,
	0x41, 0x60, 0xbe, 0x51, 0x3c, 0x17, 0x73, 0x6a, 0x45, 0x89, 0x90, 0x0c, 0x92, 0x9d, 0x18, 0x96,
	0x48, 0x2c, 0x79, 0xbf, 0xf3, 0x2b, 0xa3, 0x02, 0xa3, 0x06, 0xb7, 0x91, 0x36, 0x09, 0x6e, 0x0b,
	0x02, 0x99, 0xec, 0x94, 0x77, 0xe1, 0xa1, 0x1c, 0xc3, 0x8d, 0x87, 0x72, 0x0c, 0x1f, 0x1e, 0xca,
	0x31, 0x36, 0x3c, 0x92, 0x63, 0x5c, 0xf1, 0x48, 0x8e, 0xf1, 0xc4, 0x23, 0x39, 0xc6, 0x0b, 0x8f,
	0xe4, 0x18, 0x1f, 0x3c, 0x92, 0x63, 0x7c, 0xf1, 0x48, 0x8e, 0xe1, 0xc3, 0x23, 0x39, 0xc6, 0x09,
	0x8f, 0xe5, 0x18, 0x2e, 0x3c, 0x96, 0x63, 0xb8, 0xf1, 0x58, 0x8e, 0x21, 0xca, 0x22, 0x3d, 0xb3,
	0x24, 0xa3, 0x34, 0x49, 0x2f, 0x39, 0x3f, 0x57, 0x3f, 0x3d, 0x3f, 0x3f, 0x3d, 0x27, 0x55, 0x1f,
	0x11, 0xec, 0xf8, 0xe2, 0x12, 0x10, 0x00, 0x00, 0xff, 0xff, 0x64, 0x9b, 0xac, 0xcf, 0xea, 0x01,
	0x00, 0x00,
}

func (this *TensorsReadbackCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*TensorsReadbackCalculatorOptions)
	if !ok {
		that2, ok := that.(TensorsReadbackCalculatorOptions)
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
	if len(this.TensorShape) != len(that1.TensorShape) {
		return false
	}
	for i := range this.TensorShape {
		if !this.TensorShape[i].Equal(that1.TensorShape[i]) {
			return false
		}
	}
	return true
}
func (this *TensorsReadbackCalculatorOptions_TensorShape) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*TensorsReadbackCalculatorOptions_TensorShape)
	if !ok {
		that2, ok := that.(TensorsReadbackCalculatorOptions_TensorShape)
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
	if len(this.Dims) != len(that1.Dims) {
		return false
	}
	for i := range this.Dims {
		if this.Dims[i] != that1.Dims[i] {
			return false
		}
	}
	return true
}
func (this *TensorsReadbackCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&tensor.TensorsReadbackCalculatorOptions{")
	if this.TensorShape != nil {
		s = append(s, "TensorShape: "+fmt.Sprintf("%#v", this.TensorShape)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *TensorsReadbackCalculatorOptions_TensorShape) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&tensor.TensorsReadbackCalculatorOptions_TensorShape{")
	if this.Dims != nil {
		s = append(s, "Dims: "+fmt.Sprintf("%#v", this.Dims)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringTensorsReadbackCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *TensorsReadbackCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TensorsReadbackCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *TensorsReadbackCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.TensorShape) > 0 {
		for iNdEx := len(m.TensorShape) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.TensorShape[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintTensorsReadbackCalculator(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func (m *TensorsReadbackCalculatorOptions_TensorShape) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *TensorsReadbackCalculatorOptions_TensorShape) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *TensorsReadbackCalculatorOptions_TensorShape) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Dims) > 0 {
		dAtA2 := make([]byte, len(m.Dims)*10)
		var j1 int
		for _, num1 := range m.Dims {
			num := uint64(num1)
			for num >= 1<<7 {
				dAtA2[j1] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j1++
			}
			dAtA2[j1] = uint8(num)
			j1++
		}
		i -= j1
		copy(dAtA[i:], dAtA2[:j1])
		i = encodeVarintTensorsReadbackCalculator(dAtA, i, uint64(j1))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintTensorsReadbackCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovTensorsReadbackCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *TensorsReadbackCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.TensorShape) > 0 {
		for _, e := range m.TensorShape {
			l = e.Size()
			n += 1 + l + sovTensorsReadbackCalculator(uint64(l))
		}
	}
	return n
}

func (m *TensorsReadbackCalculatorOptions_TensorShape) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Dims) > 0 {
		l = 0
		for _, e := range m.Dims {
			l += sovTensorsReadbackCalculator(uint64(e))
		}
		n += 1 + sovTensorsReadbackCalculator(uint64(l)) + l
	}
	return n
}

func sovTensorsReadbackCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozTensorsReadbackCalculator(x uint64) (n int) {
	return sovTensorsReadbackCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *TensorsReadbackCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForTensorShape := "[]*TensorsReadbackCalculatorOptions_TensorShape{"
	for _, f := range this.TensorShape {
		repeatedStringForTensorShape += strings.Replace(fmt.Sprintf("%v", f), "TensorsReadbackCalculatorOptions_TensorShape", "TensorsReadbackCalculatorOptions_TensorShape", 1) + ","
	}
	repeatedStringForTensorShape += "}"
	s := strings.Join([]string{`&TensorsReadbackCalculatorOptions{`,
		`TensorShape:` + repeatedStringForTensorShape + `,`,
		`}`,
	}, "")
	return s
}
func (this *TensorsReadbackCalculatorOptions_TensorShape) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&TensorsReadbackCalculatorOptions_TensorShape{`,
		`Dims:` + fmt.Sprintf("%v", this.Dims) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringTensorsReadbackCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *TensorsReadbackCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorsReadbackCalculator
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
			return fmt.Errorf("proto: TensorsReadbackCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TensorsReadbackCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field TensorShape", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowTensorsReadbackCalculator
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
				return ErrInvalidLengthTensorsReadbackCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthTensorsReadbackCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.TensorShape = append(m.TensorShape, &TensorsReadbackCalculatorOptions_TensorShape{})
			if err := m.TensorShape[len(m.TensorShape)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipTensorsReadbackCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTensorsReadbackCalculator
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
func (m *TensorsReadbackCalculatorOptions_TensorShape) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowTensorsReadbackCalculator
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
			return fmt.Errorf("proto: TensorShape: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: TensorShape: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType == 0 {
				var v int32
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowTensorsReadbackCalculator
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
				m.Dims = append(m.Dims, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowTensorsReadbackCalculator
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
					return ErrInvalidLengthTensorsReadbackCalculator
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthTensorsReadbackCalculator
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
				if elementCount != 0 && len(m.Dims) == 0 {
					m.Dims = make([]int32, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v int32
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowTensorsReadbackCalculator
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
					m.Dims = append(m.Dims, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field Dims", wireType)
			}
		default:
			iNdEx = preIndex
			skippy, err := skipTensorsReadbackCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthTensorsReadbackCalculator
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
func skipTensorsReadbackCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowTensorsReadbackCalculator
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
					return 0, ErrIntOverflowTensorsReadbackCalculator
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
					return 0, ErrIntOverflowTensorsReadbackCalculator
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
				return 0, ErrInvalidLengthTensorsReadbackCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupTensorsReadbackCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthTensorsReadbackCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthTensorsReadbackCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowTensorsReadbackCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupTensorsReadbackCalculator = fmt.Errorf("proto: unexpected end of group")
)
