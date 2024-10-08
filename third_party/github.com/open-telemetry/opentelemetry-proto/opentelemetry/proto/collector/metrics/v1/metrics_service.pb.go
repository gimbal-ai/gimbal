// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: opentelemetry/proto/collector/metrics/v1/metrics_service.proto

package v1

import (
	context "context"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	v1 "go.opentelemetry.io/proto/otlp/metrics/v1"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
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

type ExportMetricsServiceRequest struct {
	ResourceMetrics []*v1.ResourceMetrics `protobuf:"bytes,1,rep,name=resource_metrics,json=resourceMetrics,proto3" json:"resource_metrics,omitempty"`
}

func (m *ExportMetricsServiceRequest) Reset()      { *m = ExportMetricsServiceRequest{} }
func (*ExportMetricsServiceRequest) ProtoMessage() {}
func (*ExportMetricsServiceRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_75fb6015e6e64798, []int{0}
}
func (m *ExportMetricsServiceRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ExportMetricsServiceRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ExportMetricsServiceRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ExportMetricsServiceRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ExportMetricsServiceRequest.Merge(m, src)
}
func (m *ExportMetricsServiceRequest) XXX_Size() int {
	return m.Size()
}
func (m *ExportMetricsServiceRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_ExportMetricsServiceRequest.DiscardUnknown(m)
}

var xxx_messageInfo_ExportMetricsServiceRequest proto.InternalMessageInfo

func (m *ExportMetricsServiceRequest) GetResourceMetrics() []*v1.ResourceMetrics {
	if m != nil {
		return m.ResourceMetrics
	}
	return nil
}

type ExportMetricsServiceResponse struct {
	PartialSuccess *ExportMetricsPartialSuccess `protobuf:"bytes,1,opt,name=partial_success,json=partialSuccess,proto3" json:"partial_success,omitempty"`
}

func (m *ExportMetricsServiceResponse) Reset()      { *m = ExportMetricsServiceResponse{} }
func (*ExportMetricsServiceResponse) ProtoMessage() {}
func (*ExportMetricsServiceResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_75fb6015e6e64798, []int{1}
}
func (m *ExportMetricsServiceResponse) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ExportMetricsServiceResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ExportMetricsServiceResponse.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ExportMetricsServiceResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ExportMetricsServiceResponse.Merge(m, src)
}
func (m *ExportMetricsServiceResponse) XXX_Size() int {
	return m.Size()
}
func (m *ExportMetricsServiceResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ExportMetricsServiceResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ExportMetricsServiceResponse proto.InternalMessageInfo

func (m *ExportMetricsServiceResponse) GetPartialSuccess() *ExportMetricsPartialSuccess {
	if m != nil {
		return m.PartialSuccess
	}
	return nil
}

type ExportMetricsPartialSuccess struct {
	RejectedDataPoints int64  `protobuf:"varint,1,opt,name=rejected_data_points,json=rejectedDataPoints,proto3" json:"rejected_data_points,omitempty"`
	ErrorMessage       string `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (m *ExportMetricsPartialSuccess) Reset()      { *m = ExportMetricsPartialSuccess{} }
func (*ExportMetricsPartialSuccess) ProtoMessage() {}
func (*ExportMetricsPartialSuccess) Descriptor() ([]byte, []int) {
	return fileDescriptor_75fb6015e6e64798, []int{2}
}
func (m *ExportMetricsPartialSuccess) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ExportMetricsPartialSuccess) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ExportMetricsPartialSuccess.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ExportMetricsPartialSuccess) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ExportMetricsPartialSuccess.Merge(m, src)
}
func (m *ExportMetricsPartialSuccess) XXX_Size() int {
	return m.Size()
}
func (m *ExportMetricsPartialSuccess) XXX_DiscardUnknown() {
	xxx_messageInfo_ExportMetricsPartialSuccess.DiscardUnknown(m)
}

var xxx_messageInfo_ExportMetricsPartialSuccess proto.InternalMessageInfo

func (m *ExportMetricsPartialSuccess) GetRejectedDataPoints() int64 {
	if m != nil {
		return m.RejectedDataPoints
	}
	return 0
}

func (m *ExportMetricsPartialSuccess) GetErrorMessage() string {
	if m != nil {
		return m.ErrorMessage
	}
	return ""
}

func init() {
	proto.RegisterType((*ExportMetricsServiceRequest)(nil), "opentelemetry.proto.collector.metrics.v1.ExportMetricsServiceRequest")
	proto.RegisterType((*ExportMetricsServiceResponse)(nil), "opentelemetry.proto.collector.metrics.v1.ExportMetricsServiceResponse")
	proto.RegisterType((*ExportMetricsPartialSuccess)(nil), "opentelemetry.proto.collector.metrics.v1.ExportMetricsPartialSuccess")
}

func init() {
	proto.RegisterFile("opentelemetry/proto/collector/metrics/v1/metrics_service.proto", fileDescriptor_75fb6015e6e64798)
}

var fileDescriptor_75fb6015e6e64798 = []byte{
	// 426 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xa4, 0x93, 0xb1, 0x8e, 0xd3, 0x30,
	0x1c, 0xc6, 0xe3, 0x3b, 0xe9, 0x24, 0x7c, 0x70, 0x87, 0x0c, 0xc3, 0xe9, 0x40, 0xd6, 0x29, 0x2c,
	0x91, 0x40, 0x0e, 0x6d, 0x77, 0x86, 0x42, 0xd9, 0x2a, 0xa2, 0x14, 0x31, 0x74, 0x89, 0x82, 0xfb,
	0x57, 0x15, 0x94, 0xc6, 0xc6, 0x76, 0x2b, 0xba, 0xf1, 0x04, 0x88, 0x95, 0x37, 0x40, 0x88, 0x07,
	0x61, 0xec, 0xd8, 0x91, 0xa6, 0x0b, 0x63, 0x1f, 0x01, 0x35, 0x4e, 0x8b, 0x5c, 0x32, 0x54, 0xb0,
	0xb5, 0x9f, 0xff, 0xdf, 0xef, 0xfb, 0xf2, 0x77, 0x82, 0x9f, 0x09, 0x09, 0x85, 0x81, 0x1c, 0x26,
	0x60, 0xd4, 0x3c, 0x94, 0x4a, 0x18, 0x11, 0x72, 0x91, 0xe7, 0xc0, 0x8d, 0x50, 0xe1, 0x56, 0xcd,
	0xb8, 0x0e, 0x67, 0xad, 0xdd, 0xcf, 0x44, 0x83, 0x9a, 0x65, 0x1c, 0x58, 0x35, 0x4a, 0x02, 0xc7,
	0x6f, 0x45, 0xb6, 0xf7, 0xb3, 0xda, 0xc4, 0x66, 0xad, 0xeb, 0x27, 0x4d, 0x49, 0x7f, 0xf3, 0x2d,
	0xc2, 0x9f, 0xe3, 0x07, 0xbd, 0x0f, 0x52, 0x28, 0xd3, 0xb7, 0xf2, 0xc0, 0xa6, 0xc6, 0xf0, 0x7e,
	0x0a, 0xda, 0x90, 0x21, 0xbe, 0xab, 0x40, 0x8b, 0xa9, 0xe2, 0x90, 0xd4, 0xc6, 0x2b, 0x74, 0x73,
	0x1a, 0x9c, 0xb7, 0x43, 0xd6, 0xd4, 0xe8, 0x4f, 0x0f, 0x16, 0xd7, 0xbe, 0x1a, 0x1c, 0x5f, 0x2a,
	0x57, 0xf0, 0x3f, 0x21, 0xfc, 0xb0, 0x39, 0x5b, 0x4b, 0x51, 0x68, 0x20, 0x05, 0xbe, 0x94, 0xa9,
	0x32, 0x59, 0x9a, 0x27, 0x7a, 0xca, 0x39, 0xe8, 0x6d, 0x36, 0x0a, 0xce, 0xdb, 0x3d, 0x76, 0xec,
	0x36, 0x98, 0x13, 0x10, 0x59, 0xda, 0xc0, 0xc2, 0xe2, 0x0b, 0xe9, 0xfc, 0xf7, 0xcd, 0xc1, 0x2e,
	0xdc, 0x71, 0xf2, 0x14, 0xdf, 0x57, 0xf0, 0x0e, 0xb8, 0x81, 0x51, 0x32, 0x4a, 0x4d, 0x9a, 0x48,
	0x91, 0x15, 0xc6, 0x76, 0x3a, 0x8d, 0xc9, 0xee, 0xec, 0x45, 0x6a, 0xd2, 0xa8, 0x3a, 0x21, 0x8f,
	0xf0, 0x1d, 0x50, 0x4a, 0xa8, 0x64, 0x02, 0x5a, 0xa7, 0x63, 0xb8, 0x3a, 0xb9, 0x41, 0xc1, 0xad,
	0xf8, 0x76, 0x25, 0xf6, 0xad, 0xd6, 0xfe, 0x8e, 0xf0, 0x85, 0xbb, 0x00, 0xf2, 0x05, 0xe1, 0x33,
	0xdb, 0x84, 0xfc, 0xeb, 0xa3, 0xba, 0xf7, 0x78, 0xfd, 0xf2, 0x7f, 0x31, 0xf6, 0x4a, 0x7c, 0xaf,
	0xbb, 0x44, 0x8b, 0x15, 0xf5, 0x96, 0x2b, 0xea, 0x6d, 0x56, 0x14, 0x7d, 0x2c, 0x29, 0xfa, 0x5a,
	0x52, 0xf4, 0xa3, 0xa4, 0x68, 0x51, 0x52, 0xf4, 0xb3, 0xa4, 0xe8, 0x57, 0x49, 0xbd, 0x4d, 0x49,
	0xd1, 0xe7, 0x35, 0xf5, 0x16, 0x6b, 0xea, 0x2d, 0xd7, 0xd4, 0xc3, 0x8f, 0x33, 0x71, 0x74, 0x85,
	0xee, 0x3d, 0x37, 0x3d, 0xda, 0x4e, 0x46, 0x68, 0xd8, 0x19, 0x1f, 0x32, 0x32, 0x51, 0xbf, 0xdf,
	0xc2, 0xe4, 0xb2, 0xf1, 0x73, 0xfa, 0x76, 0x12, 0xbc, 0x92, 0x50, 0xbc, 0xde, 0x5b, 0x2a, 0x18,
	0x7b, 0xbe, 0x8f, 0xad, 0xa3, 0xd8, 0x9b, 0xd6, 0xdb, 0xb3, 0x8a, 0xd5, 0xf9, 0x1d, 0x00, 0x00,
	0xff, 0xff, 0xbe, 0x4e, 0x8b, 0x85, 0xac, 0x03, 0x00, 0x00,
}

func (this *ExportMetricsServiceRequest) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ExportMetricsServiceRequest)
	if !ok {
		that2, ok := that.(ExportMetricsServiceRequest)
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
	if len(this.ResourceMetrics) != len(that1.ResourceMetrics) {
		return false
	}
	for i := range this.ResourceMetrics {
		if !this.ResourceMetrics[i].Equal(that1.ResourceMetrics[i]) {
			return false
		}
	}
	return true
}
func (this *ExportMetricsServiceResponse) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ExportMetricsServiceResponse)
	if !ok {
		that2, ok := that.(ExportMetricsServiceResponse)
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
	if !this.PartialSuccess.Equal(that1.PartialSuccess) {
		return false
	}
	return true
}
func (this *ExportMetricsPartialSuccess) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ExportMetricsPartialSuccess)
	if !ok {
		that2, ok := that.(ExportMetricsPartialSuccess)
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
	if this.RejectedDataPoints != that1.RejectedDataPoints {
		return false
	}
	if this.ErrorMessage != that1.ErrorMessage {
		return false
	}
	return true
}
func (this *ExportMetricsServiceRequest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&v1.ExportMetricsServiceRequest{")
	if this.ResourceMetrics != nil {
		s = append(s, "ResourceMetrics: "+fmt.Sprintf("%#v", this.ResourceMetrics)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *ExportMetricsServiceResponse) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&v1.ExportMetricsServiceResponse{")
	if this.PartialSuccess != nil {
		s = append(s, "PartialSuccess: "+fmt.Sprintf("%#v", this.PartialSuccess)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *ExportMetricsPartialSuccess) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&v1.ExportMetricsPartialSuccess{")
	s = append(s, "RejectedDataPoints: "+fmt.Sprintf("%#v", this.RejectedDataPoints)+",\n")
	s = append(s, "ErrorMessage: "+fmt.Sprintf("%#v", this.ErrorMessage)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringMetricsService(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// MetricsServiceClient is the client API for MetricsService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type MetricsServiceClient interface {
	Export(ctx context.Context, in *ExportMetricsServiceRequest, opts ...grpc.CallOption) (*ExportMetricsServiceResponse, error)
}

type metricsServiceClient struct {
	cc *grpc.ClientConn
}

func NewMetricsServiceClient(cc *grpc.ClientConn) MetricsServiceClient {
	return &metricsServiceClient{cc}
}

func (c *metricsServiceClient) Export(ctx context.Context, in *ExportMetricsServiceRequest, opts ...grpc.CallOption) (*ExportMetricsServiceResponse, error) {
	out := new(ExportMetricsServiceResponse)
	err := c.cc.Invoke(ctx, "/opentelemetry.proto.collector.metrics.v1.MetricsService/Export", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// MetricsServiceServer is the server API for MetricsService service.
type MetricsServiceServer interface {
	Export(context.Context, *ExportMetricsServiceRequest) (*ExportMetricsServiceResponse, error)
}

// UnimplementedMetricsServiceServer can be embedded to have forward compatible implementations.
type UnimplementedMetricsServiceServer struct {
}

func (*UnimplementedMetricsServiceServer) Export(ctx context.Context, req *ExportMetricsServiceRequest) (*ExportMetricsServiceResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Export not implemented")
}

func RegisterMetricsServiceServer(s *grpc.Server, srv MetricsServiceServer) {
	s.RegisterService(&_MetricsService_serviceDesc, srv)
}

func _MetricsService_Export_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ExportMetricsServiceRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(MetricsServiceServer).Export(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/opentelemetry.proto.collector.metrics.v1.MetricsService/Export",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(MetricsServiceServer).Export(ctx, req.(*ExportMetricsServiceRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _MetricsService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "opentelemetry.proto.collector.metrics.v1.MetricsService",
	HandlerType: (*MetricsServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Export",
			Handler:    _MetricsService_Export_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "opentelemetry/proto/collector/metrics/v1/metrics_service.proto",
}

func (m *ExportMetricsServiceRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ExportMetricsServiceRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ExportMetricsServiceRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.ResourceMetrics) > 0 {
		for iNdEx := len(m.ResourceMetrics) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.ResourceMetrics[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintMetricsService(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func (m *ExportMetricsServiceResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ExportMetricsServiceResponse) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ExportMetricsServiceResponse) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.PartialSuccess != nil {
		{
			size, err := m.PartialSuccess.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintMetricsService(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *ExportMetricsPartialSuccess) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ExportMetricsPartialSuccess) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ExportMetricsPartialSuccess) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.ErrorMessage) > 0 {
		i -= len(m.ErrorMessage)
		copy(dAtA[i:], m.ErrorMessage)
		i = encodeVarintMetricsService(dAtA, i, uint64(len(m.ErrorMessage)))
		i--
		dAtA[i] = 0x12
	}
	if m.RejectedDataPoints != 0 {
		i = encodeVarintMetricsService(dAtA, i, uint64(m.RejectedDataPoints))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintMetricsService(dAtA []byte, offset int, v uint64) int {
	offset -= sovMetricsService(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ExportMetricsServiceRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.ResourceMetrics) > 0 {
		for _, e := range m.ResourceMetrics {
			l = e.Size()
			n += 1 + l + sovMetricsService(uint64(l))
		}
	}
	return n
}

func (m *ExportMetricsServiceResponse) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.PartialSuccess != nil {
		l = m.PartialSuccess.Size()
		n += 1 + l + sovMetricsService(uint64(l))
	}
	return n
}

func (m *ExportMetricsPartialSuccess) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.RejectedDataPoints != 0 {
		n += 1 + sovMetricsService(uint64(m.RejectedDataPoints))
	}
	l = len(m.ErrorMessage)
	if l > 0 {
		n += 1 + l + sovMetricsService(uint64(l))
	}
	return n
}

func sovMetricsService(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozMetricsService(x uint64) (n int) {
	return sovMetricsService(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ExportMetricsServiceRequest) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForResourceMetrics := "[]*ResourceMetrics{"
	for _, f := range this.ResourceMetrics {
		repeatedStringForResourceMetrics += strings.Replace(fmt.Sprintf("%v", f), "ResourceMetrics", "v1.ResourceMetrics", 1) + ","
	}
	repeatedStringForResourceMetrics += "}"
	s := strings.Join([]string{`&ExportMetricsServiceRequest{`,
		`ResourceMetrics:` + repeatedStringForResourceMetrics + `,`,
		`}`,
	}, "")
	return s
}
func (this *ExportMetricsServiceResponse) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ExportMetricsServiceResponse{`,
		`PartialSuccess:` + strings.Replace(this.PartialSuccess.String(), "ExportMetricsPartialSuccess", "ExportMetricsPartialSuccess", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *ExportMetricsPartialSuccess) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ExportMetricsPartialSuccess{`,
		`RejectedDataPoints:` + fmt.Sprintf("%v", this.RejectedDataPoints) + `,`,
		`ErrorMessage:` + fmt.Sprintf("%v", this.ErrorMessage) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringMetricsService(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ExportMetricsServiceRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMetricsService
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
			return fmt.Errorf("proto: ExportMetricsServiceRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExportMetricsServiceRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ResourceMetrics", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMetricsService
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
				return ErrInvalidLengthMetricsService
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMetricsService
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ResourceMetrics = append(m.ResourceMetrics, &v1.ResourceMetrics{})
			if err := m.ResourceMetrics[len(m.ResourceMetrics)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipMetricsService(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMetricsService
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
func (m *ExportMetricsServiceResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMetricsService
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
			return fmt.Errorf("proto: ExportMetricsServiceResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExportMetricsServiceResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field PartialSuccess", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMetricsService
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
				return ErrInvalidLengthMetricsService
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMetricsService
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.PartialSuccess == nil {
				m.PartialSuccess = &ExportMetricsPartialSuccess{}
			}
			if err := m.PartialSuccess.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipMetricsService(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMetricsService
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
func (m *ExportMetricsPartialSuccess) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMetricsService
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
			return fmt.Errorf("proto: ExportMetricsPartialSuccess: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExportMetricsPartialSuccess: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RejectedDataPoints", wireType)
			}
			m.RejectedDataPoints = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMetricsService
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.RejectedDataPoints |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ErrorMessage", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMetricsService
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
				return ErrInvalidLengthMetricsService
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthMetricsService
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ErrorMessage = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipMetricsService(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMetricsService
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
func skipMetricsService(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowMetricsService
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
					return 0, ErrIntOverflowMetricsService
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
					return 0, ErrIntOverflowMetricsService
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
				return 0, ErrInvalidLengthMetricsService
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupMetricsService
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthMetricsService
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthMetricsService        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowMetricsService          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupMetricsService = fmt.Errorf("proto: unexpected end of group")
)
