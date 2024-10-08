// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/thread_pool_executor.proto

package framework

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

type ThreadPoolExecutorOptions_ProcessorPerformance int32

const (
	PROCESSOR_PERFORMANCE_NORMAL ThreadPoolExecutorOptions_ProcessorPerformance = 0
	PROCESSOR_PERFORMANCE_LOW    ThreadPoolExecutorOptions_ProcessorPerformance = 1
	PROCESSOR_PERFORMANCE_HIGH   ThreadPoolExecutorOptions_ProcessorPerformance = 2
)

var ThreadPoolExecutorOptions_ProcessorPerformance_name = map[int32]string{
	0: "PROCESSOR_PERFORMANCE_NORMAL",
	1: "PROCESSOR_PERFORMANCE_LOW",
	2: "PROCESSOR_PERFORMANCE_HIGH",
}

var ThreadPoolExecutorOptions_ProcessorPerformance_value = map[string]int32{
	"PROCESSOR_PERFORMANCE_NORMAL": 0,
	"PROCESSOR_PERFORMANCE_LOW":    1,
	"PROCESSOR_PERFORMANCE_HIGH":   2,
}

func (x ThreadPoolExecutorOptions_ProcessorPerformance) Enum() *ThreadPoolExecutorOptions_ProcessorPerformance {
	p := new(ThreadPoolExecutorOptions_ProcessorPerformance)
	*p = x
	return p
}

func (x ThreadPoolExecutorOptions_ProcessorPerformance) MarshalJSON() ([]byte, error) {
	return proto.MarshalJSONEnum(ThreadPoolExecutorOptions_ProcessorPerformance_name, int32(x))
}

func (x *ThreadPoolExecutorOptions_ProcessorPerformance) UnmarshalJSON(data []byte) error {
	value, err := proto.UnmarshalJSONEnum(ThreadPoolExecutorOptions_ProcessorPerformance_value, data, "ThreadPoolExecutorOptions_ProcessorPerformance")
	if err != nil {
		return err
	}
	*x = ThreadPoolExecutorOptions_ProcessorPerformance(value)
	return nil
}

func (ThreadPoolExecutorOptions_ProcessorPerformance) EnumDescriptor() ([]byte, []int) {
	return fileDescriptor_76ae72218d0a3598, []int{0, 0}
}

type ThreadPoolExecutorOptions struct {
	NumThreads                  int32                                          `protobuf:"varint,1,opt,name=num_threads,json=numThreads" json:"num_threads"`
	StackSize                   int32                                          `protobuf:"varint,2,opt,name=stack_size,json=stackSize" json:"stack_size"`
	NicePriorityLevel           int32                                          `protobuf:"varint,3,opt,name=nice_priority_level,json=nicePriorityLevel" json:"nice_priority_level"`
	RequireProcessorPerformance ThreadPoolExecutorOptions_ProcessorPerformance `protobuf:"varint,4,opt,name=require_processor_performance,json=requireProcessorPerformance,enum=mediapipe.ThreadPoolExecutorOptions_ProcessorPerformance" json:"require_processor_performance"`
	ThreadNamePrefix            string                                         `protobuf:"bytes,5,opt,name=thread_name_prefix,json=threadNamePrefix" json:"thread_name_prefix"`
}

func (m *ThreadPoolExecutorOptions) Reset()      { *m = ThreadPoolExecutorOptions{} }
func (*ThreadPoolExecutorOptions) ProtoMessage() {}
func (*ThreadPoolExecutorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_76ae72218d0a3598, []int{0}
}
func (m *ThreadPoolExecutorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ThreadPoolExecutorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ThreadPoolExecutorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ThreadPoolExecutorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ThreadPoolExecutorOptions.Merge(m, src)
}
func (m *ThreadPoolExecutorOptions) XXX_Size() int {
	return m.Size()
}
func (m *ThreadPoolExecutorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_ThreadPoolExecutorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_ThreadPoolExecutorOptions proto.InternalMessageInfo

func (m *ThreadPoolExecutorOptions) GetNumThreads() int32 {
	if m != nil {
		return m.NumThreads
	}
	return 0
}

func (m *ThreadPoolExecutorOptions) GetStackSize() int32 {
	if m != nil {
		return m.StackSize
	}
	return 0
}

func (m *ThreadPoolExecutorOptions) GetNicePriorityLevel() int32 {
	if m != nil {
		return m.NicePriorityLevel
	}
	return 0
}

func (m *ThreadPoolExecutorOptions) GetRequireProcessorPerformance() ThreadPoolExecutorOptions_ProcessorPerformance {
	if m != nil {
		return m.RequireProcessorPerformance
	}
	return PROCESSOR_PERFORMANCE_NORMAL
}

func (m *ThreadPoolExecutorOptions) GetThreadNamePrefix() string {
	if m != nil {
		return m.ThreadNamePrefix
	}
	return ""
}

var E_ThreadPoolExecutorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*MediaPipeOptions)(nil),
	ExtensionType: (*ThreadPoolExecutorOptions)(nil),
	Field:         157116819,
	Name:          "mediapipe.ThreadPoolExecutorOptions.ext",
	Tag:           "bytes,157116819,opt,name=ext",
	Filename:      "mediapipe/framework/thread_pool_executor.proto",
}

func init() {
	proto.RegisterEnum("mediapipe.ThreadPoolExecutorOptions_ProcessorPerformance", ThreadPoolExecutorOptions_ProcessorPerformance_name, ThreadPoolExecutorOptions_ProcessorPerformance_value)
	proto.RegisterExtension(E_ThreadPoolExecutorOptions_Ext)
	proto.RegisterType((*ThreadPoolExecutorOptions)(nil), "mediapipe.ThreadPoolExecutorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/framework/thread_pool_executor.proto", fileDescriptor_76ae72218d0a3598)
}

var fileDescriptor_76ae72218d0a3598 = []byte{
	// 462 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x8c, 0x92, 0xc1, 0x6b, 0xd4, 0x40,
	0x14, 0xc6, 0x33, 0xed, 0xf6, 0xb0, 0x53, 0x90, 0x75, 0xf4, 0x90, 0xb6, 0x76, 0x0c, 0x55, 0x61,
	0x41, 0x48, 0x60, 0xf1, 0xe2, 0xd1, 0x96, 0x68, 0x95, 0xed, 0x26, 0x64, 0x45, 0xc1, 0xcb, 0x10,
	0xd3, 0xb7, 0xdb, 0xa1, 0x49, 0x5e, 0x9c, 0x24, 0x76, 0xed, 0x49, 0xf0, 0x1f, 0x10, 0xfc, 0x27,
	0xfa, 0xa7, 0xf4, 0xb8, 0xe0, 0xa5, 0x27, 0x71, 0xb3, 0x17, 0x8f, 0xbd, 0x78, 0x97, 0x6c, 0x62,
	0x77, 0x0f, 0x11, 0xbc, 0x0d, 0xef, 0xfb, 0x7d, 0xdf, 0x4b, 0x3e, 0x1e, 0x35, 0x23, 0x38, 0x96,
	0x7e, 0x22, 0x13, 0xb0, 0x46, 0xca, 0x8f, 0xe0, 0x0c, 0xd5, 0xa9, 0x95, 0x9d, 0x28, 0xf0, 0x8f,
	0x45, 0x82, 0x18, 0x0a, 0x98, 0x40, 0x90, 0x67, 0xa8, 0xcc, 0x44, 0x61, 0x86, 0xac, 0x7d, 0xc3,
	0x6f, 0x3f, 0x6e, 0xb2, 0xde, 0xcc, 0x04, 0x26, 0x99, 0xc4, 0x38, 0xad, 0x7c, 0x7b, 0x17, 0x2d,
	0xba, 0xf5, 0x7a, 0x11, 0xeb, 0x22, 0x86, 0x76, 0x1d, 0xea, 0x54, 0x0c, 0x7b, 0x44, 0x37, 0xe3,
	0x3c, 0x12, 0xd5, 0xde, 0x54, 0x27, 0x06, 0xe9, 0x6e, 0xec, 0xb7, 0x2e, 0x7f, 0xdc, 0xd7, 0x3c,
	0x1a, 0xe7, 0x51, 0x65, 0x4c, 0xd9, 0x03, 0x4a, 0xd3, 0xcc, 0x0f, 0x4e, 0x45, 0x2a, 0xcf, 0x41,
	0x5f, 0x5b, 0xa1, 0xda, 0x8b, 0xf9, 0x50, 0x9e, 0x03, 0x7b, 0x42, 0xef, 0xc4, 0x32, 0x00, 0x91,
	0x28, 0x89, 0x4a, 0x66, 0x9f, 0x44, 0x08, 0x1f, 0x21, 0xd4, 0xd7, 0x57, 0xe8, 0xdb, 0x25, 0xe0,
	0xd6, 0x7a, 0xbf, 0x94, 0xd9, 0x17, 0x42, 0x77, 0x15, 0x7c, 0xc8, 0xa5, 0x2a, 0x9d, 0x18, 0x40,
	0x9a, 0xa2, 0x12, 0x09, 0xa8, 0x11, 0xaa, 0xc8, 0x8f, 0x03, 0xd0, 0x5b, 0x06, 0xe9, 0xde, 0xea,
	0x3d, 0x5d, 0x16, 0x66, 0xfe, 0xf3, 0x7f, 0x4c, 0xf7, 0x6f, 0x82, 0xbb, 0x0c, 0xa8, 0x77, 0xef,
	0xd4, 0x5b, 0x9a, 0x10, 0xd6, 0xa3, 0xac, 0xee, 0x3e, 0xf6, 0xa3, 0xf2, 0x43, 0x60, 0x24, 0x27,
	0xfa, 0x86, 0x41, 0xba, 0xed, 0xda, 0xde, 0xa9, 0xf4, 0x81, 0x1f, 0x81, 0xbb, 0x50, 0xf7, 0xce,
	0xe8, 0xdd, 0xc6, 0x2c, 0x83, 0xde, 0x73, 0x3d, 0xe7, 0xc0, 0x1e, 0x0e, 0x1d, 0x4f, 0xb8, 0xb6,
	0xf7, 0xdc, 0xf1, 0x8e, 0x9e, 0x0d, 0x0e, 0x6c, 0x31, 0x28, 0x1f, 0xfd, 0x8e, 0xc6, 0x76, 0xe9,
	0x56, 0x33, 0xd1, 0x77, 0xde, 0x76, 0x08, 0xe3, 0x74, 0xbb, 0x59, 0x3e, 0x7c, 0xf9, 0xe2, 0xb0,
	0xb3, 0xd6, 0x7b, 0x43, 0xd7, 0x61, 0x92, 0xb1, 0x9d, 0x95, 0x46, 0x8e, 0xca, 0x97, 0x2b, 0x13,
	0xa8, 0x8b, 0xd0, 0xbf, 0x7d, 0xff, 0xfd, 0xca, 0x20, 0xdd, 0xcd, 0xde, 0xc3, 0xff, 0xe9, 0xcd,
	0x2b, 0x03, 0xf7, 0x61, 0x3a, 0xe3, 0xda, 0xd5, 0x8c, 0x6b, 0xd7, 0x33, 0x4e, 0x3e, 0x17, 0x9c,
	0x5c, 0x14, 0x9c, 0x5c, 0x16, 0x9c, 0x4c, 0x0b, 0x4e, 0x7e, 0x16, 0x9c, 0xfc, 0x2a, 0xb8, 0x76,
	0x5d, 0x70, 0xf2, 0x75, 0xce, 0xb5, 0xe9, 0x9c, 0x6b, 0x57, 0x73, 0xae, 0xbd, 0xb3, 0xc6, 0x32,
	0x3b, 0xc9, 0xdf, 0x9b, 0x01, 0x46, 0xd6, 0x18, 0x71, 0x1c, 0xc2, 0xf2, 0x16, 0xad, 0x86, 0x4b,
	0xfd, 0x13, 0x00, 0x00, 0xff, 0xff, 0xe9, 0x1c, 0x19, 0x4c, 0xfa, 0x02, 0x00, 0x00,
}

func (x ThreadPoolExecutorOptions_ProcessorPerformance) String() string {
	s, ok := ThreadPoolExecutorOptions_ProcessorPerformance_name[int32(x)]
	if ok {
		return s
	}
	return strconv.Itoa(int(x))
}
func (this *ThreadPoolExecutorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ThreadPoolExecutorOptions)
	if !ok {
		that2, ok := that.(ThreadPoolExecutorOptions)
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
	if this.NumThreads != that1.NumThreads {
		return false
	}
	if this.StackSize != that1.StackSize {
		return false
	}
	if this.NicePriorityLevel != that1.NicePriorityLevel {
		return false
	}
	if this.RequireProcessorPerformance != that1.RequireProcessorPerformance {
		return false
	}
	if this.ThreadNamePrefix != that1.ThreadNamePrefix {
		return false
	}
	return true
}
func (this *ThreadPoolExecutorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 9)
	s = append(s, "&framework.ThreadPoolExecutorOptions{")
	s = append(s, "NumThreads: "+fmt.Sprintf("%#v", this.NumThreads)+",\n")
	s = append(s, "StackSize: "+fmt.Sprintf("%#v", this.StackSize)+",\n")
	s = append(s, "NicePriorityLevel: "+fmt.Sprintf("%#v", this.NicePriorityLevel)+",\n")
	s = append(s, "RequireProcessorPerformance: "+fmt.Sprintf("%#v", this.RequireProcessorPerformance)+",\n")
	s = append(s, "ThreadNamePrefix: "+fmt.Sprintf("%#v", this.ThreadNamePrefix)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringThreadPoolExecutor(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *ThreadPoolExecutorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ThreadPoolExecutorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ThreadPoolExecutorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= len(m.ThreadNamePrefix)
	copy(dAtA[i:], m.ThreadNamePrefix)
	i = encodeVarintThreadPoolExecutor(dAtA, i, uint64(len(m.ThreadNamePrefix)))
	i--
	dAtA[i] = 0x2a
	i = encodeVarintThreadPoolExecutor(dAtA, i, uint64(m.RequireProcessorPerformance))
	i--
	dAtA[i] = 0x20
	i = encodeVarintThreadPoolExecutor(dAtA, i, uint64(m.NicePriorityLevel))
	i--
	dAtA[i] = 0x18
	i = encodeVarintThreadPoolExecutor(dAtA, i, uint64(m.StackSize))
	i--
	dAtA[i] = 0x10
	i = encodeVarintThreadPoolExecutor(dAtA, i, uint64(m.NumThreads))
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func encodeVarintThreadPoolExecutor(dAtA []byte, offset int, v uint64) int {
	offset -= sovThreadPoolExecutor(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ThreadPoolExecutorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 1 + sovThreadPoolExecutor(uint64(m.NumThreads))
	n += 1 + sovThreadPoolExecutor(uint64(m.StackSize))
	n += 1 + sovThreadPoolExecutor(uint64(m.NicePriorityLevel))
	n += 1 + sovThreadPoolExecutor(uint64(m.RequireProcessorPerformance))
	l = len(m.ThreadNamePrefix)
	n += 1 + l + sovThreadPoolExecutor(uint64(l))
	return n
}

func sovThreadPoolExecutor(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozThreadPoolExecutor(x uint64) (n int) {
	return sovThreadPoolExecutor(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ThreadPoolExecutorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ThreadPoolExecutorOptions{`,
		`NumThreads:` + fmt.Sprintf("%v", this.NumThreads) + `,`,
		`StackSize:` + fmt.Sprintf("%v", this.StackSize) + `,`,
		`NicePriorityLevel:` + fmt.Sprintf("%v", this.NicePriorityLevel) + `,`,
		`RequireProcessorPerformance:` + fmt.Sprintf("%v", this.RequireProcessorPerformance) + `,`,
		`ThreadNamePrefix:` + fmt.Sprintf("%v", this.ThreadNamePrefix) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringThreadPoolExecutor(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ThreadPoolExecutorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowThreadPoolExecutor
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
			return fmt.Errorf("proto: ThreadPoolExecutorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ThreadPoolExecutorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NumThreads", wireType)
			}
			m.NumThreads = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowThreadPoolExecutor
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.NumThreads |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field StackSize", wireType)
			}
			m.StackSize = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowThreadPoolExecutor
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.StackSize |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field NicePriorityLevel", wireType)
			}
			m.NicePriorityLevel = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowThreadPoolExecutor
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.NicePriorityLevel |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RequireProcessorPerformance", wireType)
			}
			m.RequireProcessorPerformance = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowThreadPoolExecutor
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.RequireProcessorPerformance |= ThreadPoolExecutorOptions_ProcessorPerformance(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 5:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ThreadNamePrefix", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowThreadPoolExecutor
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
				return ErrInvalidLengthThreadPoolExecutor
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthThreadPoolExecutor
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ThreadNamePrefix = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipThreadPoolExecutor(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthThreadPoolExecutor
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
func skipThreadPoolExecutor(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowThreadPoolExecutor
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
					return 0, ErrIntOverflowThreadPoolExecutor
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
					return 0, ErrIntOverflowThreadPoolExecutor
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
				return 0, ErrInvalidLengthThreadPoolExecutor
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupThreadPoolExecutor
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthThreadPoolExecutor
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthThreadPoolExecutor        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowThreadPoolExecutor          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupThreadPoolExecutor = fmt.Errorf("proto: unexpected end of group")
)
