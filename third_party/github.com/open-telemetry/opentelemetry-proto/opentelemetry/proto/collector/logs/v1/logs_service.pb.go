// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: opentelemetry/proto/collector/logs/v1/logs_service.proto

package v1

import (
	context "context"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	v1 "go.opentelemetry.io/proto/otlp/logs/v1"
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

type ExportLogsServiceRequest struct {
	ResourceLogs []*v1.ResourceLogs `protobuf:"bytes,1,rep,name=resource_logs,json=resourceLogs,proto3" json:"resource_logs,omitempty"`
}

func (m *ExportLogsServiceRequest) Reset()      { *m = ExportLogsServiceRequest{} }
func (*ExportLogsServiceRequest) ProtoMessage() {}
func (*ExportLogsServiceRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_8e3bf87aaa43acd4, []int{0}
}
func (m *ExportLogsServiceRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ExportLogsServiceRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ExportLogsServiceRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ExportLogsServiceRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ExportLogsServiceRequest.Merge(m, src)
}
func (m *ExportLogsServiceRequest) XXX_Size() int {
	return m.Size()
}
func (m *ExportLogsServiceRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_ExportLogsServiceRequest.DiscardUnknown(m)
}

var xxx_messageInfo_ExportLogsServiceRequest proto.InternalMessageInfo

func (m *ExportLogsServiceRequest) GetResourceLogs() []*v1.ResourceLogs {
	if m != nil {
		return m.ResourceLogs
	}
	return nil
}

type ExportLogsServiceResponse struct {
	PartialSuccess *ExportLogsPartialSuccess `protobuf:"bytes,1,opt,name=partial_success,json=partialSuccess,proto3" json:"partial_success,omitempty"`
}

func (m *ExportLogsServiceResponse) Reset()      { *m = ExportLogsServiceResponse{} }
func (*ExportLogsServiceResponse) ProtoMessage() {}
func (*ExportLogsServiceResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_8e3bf87aaa43acd4, []int{1}
}
func (m *ExportLogsServiceResponse) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ExportLogsServiceResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ExportLogsServiceResponse.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ExportLogsServiceResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ExportLogsServiceResponse.Merge(m, src)
}
func (m *ExportLogsServiceResponse) XXX_Size() int {
	return m.Size()
}
func (m *ExportLogsServiceResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_ExportLogsServiceResponse.DiscardUnknown(m)
}

var xxx_messageInfo_ExportLogsServiceResponse proto.InternalMessageInfo

func (m *ExportLogsServiceResponse) GetPartialSuccess() *ExportLogsPartialSuccess {
	if m != nil {
		return m.PartialSuccess
	}
	return nil
}

type ExportLogsPartialSuccess struct {
	RejectedLogRecords int64  `protobuf:"varint,1,opt,name=rejected_log_records,json=rejectedLogRecords,proto3" json:"rejected_log_records,omitempty"`
	ErrorMessage       string `protobuf:"bytes,2,opt,name=error_message,json=errorMessage,proto3" json:"error_message,omitempty"`
}

func (m *ExportLogsPartialSuccess) Reset()      { *m = ExportLogsPartialSuccess{} }
func (*ExportLogsPartialSuccess) ProtoMessage() {}
func (*ExportLogsPartialSuccess) Descriptor() ([]byte, []int) {
	return fileDescriptor_8e3bf87aaa43acd4, []int{2}
}
func (m *ExportLogsPartialSuccess) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *ExportLogsPartialSuccess) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_ExportLogsPartialSuccess.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *ExportLogsPartialSuccess) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ExportLogsPartialSuccess.Merge(m, src)
}
func (m *ExportLogsPartialSuccess) XXX_Size() int {
	return m.Size()
}
func (m *ExportLogsPartialSuccess) XXX_DiscardUnknown() {
	xxx_messageInfo_ExportLogsPartialSuccess.DiscardUnknown(m)
}

var xxx_messageInfo_ExportLogsPartialSuccess proto.InternalMessageInfo

func (m *ExportLogsPartialSuccess) GetRejectedLogRecords() int64 {
	if m != nil {
		return m.RejectedLogRecords
	}
	return 0
}

func (m *ExportLogsPartialSuccess) GetErrorMessage() string {
	if m != nil {
		return m.ErrorMessage
	}
	return ""
}

func init() {
	proto.RegisterType((*ExportLogsServiceRequest)(nil), "opentelemetry.proto.collector.logs.v1.ExportLogsServiceRequest")
	proto.RegisterType((*ExportLogsServiceResponse)(nil), "opentelemetry.proto.collector.logs.v1.ExportLogsServiceResponse")
	proto.RegisterType((*ExportLogsPartialSuccess)(nil), "opentelemetry.proto.collector.logs.v1.ExportLogsPartialSuccess")
}

func init() {
	proto.RegisterFile("opentelemetry/proto/collector/logs/v1/logs_service.proto", fileDescriptor_8e3bf87aaa43acd4)
}

var fileDescriptor_8e3bf87aaa43acd4 = []byte{
	// 426 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x9c, 0x93, 0x31, 0x8f, 0xd3, 0x30,
	0x14, 0xc7, 0xe3, 0x3b, 0xe9, 0x24, 0xdc, 0x3b, 0x40, 0x16, 0x43, 0xb9, 0xc1, 0x3a, 0x05, 0x1d,
	0x0a, 0x8b, 0x73, 0x2d, 0x0b, 0x1b, 0xe8, 0x10, 0x5b, 0x81, 0x2a, 0x45, 0x0c, 0x2c, 0x51, 0x71,
	0x9f, 0x42, 0xaa, 0xb4, 0xcf, 0xb5, 0xdd, 0x0a, 0x36, 0x16, 0x46, 0x24, 0xbe, 0x00, 0x3b, 0xe2,
	0x93, 0x30, 0x30, 0x74, 0xec, 0x48, 0xd3, 0x85, 0xb1, 0x1f, 0x01, 0x25, 0x2e, 0x25, 0x81, 0x0e,
	0xe5, 0xa6, 0xc8, 0xef, 0xbd, 0xff, 0xff, 0xf7, 0xfe, 0x49, 0x4c, 0x1f, 0xa0, 0x82, 0xb1, 0x85,
	0x0c, 0x46, 0x60, 0xf5, 0xbb, 0x50, 0x69, 0xb4, 0x18, 0x4a, 0xcc, 0x32, 0x90, 0x16, 0x75, 0x98,
	0x61, 0x62, 0xc2, 0x59, 0xab, 0x7c, 0xc6, 0x06, 0xf4, 0x2c, 0x95, 0x20, 0xca, 0x21, 0x76, 0x5e,
	0x53, 0xba, 0xa2, 0xd8, 0x2a, 0x45, 0xa1, 0x10, 0xb3, 0xd6, 0xe9, 0xdd, 0x5d, 0x80, 0xaa, 0xad,
	0x53, 0xfa, 0x43, 0xda, 0x7c, 0xf2, 0x56, 0xa1, 0xb6, 0x1d, 0x4c, 0x4c, 0xcf, 0x91, 0x22, 0x98,
	0x4c, 0xc1, 0x58, 0xf6, 0x8c, 0x9e, 0x68, 0x30, 0x38, 0xd5, 0x12, 0xe2, 0x42, 0xd2, 0x24, 0x67,
	0x87, 0x41, 0xa3, 0x7d, 0x4f, 0xec, 0x5a, 0x61, 0x03, 0x16, 0xd1, 0x46, 0x51, 0xf8, 0x45, 0xc7,
	0xba, 0x72, 0xf2, 0x3f, 0x10, 0x7a, 0x7b, 0x07, 0xcc, 0x28, 0x1c, 0x1b, 0x60, 0x6f, 0xe8, 0x0d,
	0xd5, 0xd7, 0x36, 0xed, 0x67, 0xb1, 0x99, 0x4a, 0x09, 0xa6, 0xe0, 0x91, 0xa0, 0xd1, 0x7e, 0x28,
	0xf6, 0x8a, 0x2c, 0xfe, 0x58, 0x77, 0x9d, 0x4f, 0xcf, 0xd9, 0x44, 0xd7, 0x55, 0xed, 0xec, 0x4f,
	0xaa, 0x99, 0xeb, 0xb3, 0xec, 0x82, 0xde, 0xd2, 0x30, 0x04, 0x69, 0x61, 0x50, 0x64, 0x8e, 0x35,
	0x48, 0xd4, 0x03, 0xb7, 0xca, 0x61, 0xc4, 0x7e, 0xf7, 0x3a, 0x98, 0x44, 0xae, 0xc3, 0xee, 0xd0,
	0x13, 0xd0, 0x1a, 0x75, 0x3c, 0x02, 0x63, 0xfa, 0x09, 0x34, 0x0f, 0xce, 0x48, 0x70, 0x2d, 0x3a,
	0x2e, 0x8b, 0x4f, 0x5d, 0xad, 0xfd, 0x99, 0xd0, 0x46, 0x25, 0x34, 0xfb, 0x48, 0xe8, 0x91, 0xdb,
	0x81, 0xfd, 0x7f, 0xbc, 0xfa, 0x67, 0x3a, 0x7d, 0x74, 0x75, 0x03, 0xf7, 0xea, 0x7d, 0xef, 0xf2,
	0x3b, 0x99, 0x2f, 0xb9, 0xb7, 0x58, 0x72, 0x6f, 0xbd, 0xe4, 0xe4, 0x7d, 0xce, 0xc9, 0x97, 0x9c,
	0x93, 0x6f, 0x39, 0x27, 0xf3, 0x9c, 0x93, 0x1f, 0x39, 0x27, 0x3f, 0x73, 0xee, 0xad, 0x73, 0x4e,
	0x3e, 0xad, 0xb8, 0x37, 0x5f, 0x71, 0x6f, 0xb1, 0xe2, 0x1e, 0x0d, 0x52, 0xdc, 0x0f, 0x7e, 0x79,
	0xb3, 0xc2, 0xed, 0x16, 0x33, 0x5d, 0xf2, 0xea, 0x22, 0xf9, 0x5b, 0x9d, 0xe2, 0xe6, 0x4f, 0x45,
	0x9b, 0xa9, 0x7f, 0xef, 0xc3, 0xd7, 0x83, 0xf3, 0xe7, 0x0a, 0xc6, 0x2f, 0xb6, 0xf3, 0xa5, 0x93,
	0x78, 0xbc, 0xa5, 0x15, 0x10, 0xf1, 0xb2, 0xf5, 0xfa, 0xa8, 0x74, 0xb9, 0xff, 0x2b, 0x00, 0x00,
	0xff, 0xff, 0x23, 0x51, 0xa2, 0xcb, 0x67, 0x03, 0x00, 0x00,
}

func (this *ExportLogsServiceRequest) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ExportLogsServiceRequest)
	if !ok {
		that2, ok := that.(ExportLogsServiceRequest)
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
	if len(this.ResourceLogs) != len(that1.ResourceLogs) {
		return false
	}
	for i := range this.ResourceLogs {
		if !this.ResourceLogs[i].Equal(that1.ResourceLogs[i]) {
			return false
		}
	}
	return true
}
func (this *ExportLogsServiceResponse) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ExportLogsServiceResponse)
	if !ok {
		that2, ok := that.(ExportLogsServiceResponse)
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
func (this *ExportLogsPartialSuccess) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*ExportLogsPartialSuccess)
	if !ok {
		that2, ok := that.(ExportLogsPartialSuccess)
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
	if this.RejectedLogRecords != that1.RejectedLogRecords {
		return false
	}
	if this.ErrorMessage != that1.ErrorMessage {
		return false
	}
	return true
}
func (this *ExportLogsServiceRequest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&v1.ExportLogsServiceRequest{")
	if this.ResourceLogs != nil {
		s = append(s, "ResourceLogs: "+fmt.Sprintf("%#v", this.ResourceLogs)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *ExportLogsServiceResponse) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&v1.ExportLogsServiceResponse{")
	if this.PartialSuccess != nil {
		s = append(s, "PartialSuccess: "+fmt.Sprintf("%#v", this.PartialSuccess)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *ExportLogsPartialSuccess) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&v1.ExportLogsPartialSuccess{")
	s = append(s, "RejectedLogRecords: "+fmt.Sprintf("%#v", this.RejectedLogRecords)+",\n")
	s = append(s, "ErrorMessage: "+fmt.Sprintf("%#v", this.ErrorMessage)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringLogsService(v interface{}, typ string) string {
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

// LogsServiceClient is the client API for LogsService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type LogsServiceClient interface {
	Export(ctx context.Context, in *ExportLogsServiceRequest, opts ...grpc.CallOption) (*ExportLogsServiceResponse, error)
}

type logsServiceClient struct {
	cc *grpc.ClientConn
}

func NewLogsServiceClient(cc *grpc.ClientConn) LogsServiceClient {
	return &logsServiceClient{cc}
}

func (c *logsServiceClient) Export(ctx context.Context, in *ExportLogsServiceRequest, opts ...grpc.CallOption) (*ExportLogsServiceResponse, error) {
	out := new(ExportLogsServiceResponse)
	err := c.cc.Invoke(ctx, "/opentelemetry.proto.collector.logs.v1.LogsService/Export", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// LogsServiceServer is the server API for LogsService service.
type LogsServiceServer interface {
	Export(context.Context, *ExportLogsServiceRequest) (*ExportLogsServiceResponse, error)
}

// UnimplementedLogsServiceServer can be embedded to have forward compatible implementations.
type UnimplementedLogsServiceServer struct {
}

func (*UnimplementedLogsServiceServer) Export(ctx context.Context, req *ExportLogsServiceRequest) (*ExportLogsServiceResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Export not implemented")
}

func RegisterLogsServiceServer(s *grpc.Server, srv LogsServiceServer) {
	s.RegisterService(&_LogsService_serviceDesc, srv)
}

func _LogsService_Export_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ExportLogsServiceRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(LogsServiceServer).Export(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/opentelemetry.proto.collector.logs.v1.LogsService/Export",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(LogsServiceServer).Export(ctx, req.(*ExportLogsServiceRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _LogsService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "opentelemetry.proto.collector.logs.v1.LogsService",
	HandlerType: (*LogsServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Export",
			Handler:    _LogsService_Export_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "opentelemetry/proto/collector/logs/v1/logs_service.proto",
}

func (m *ExportLogsServiceRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ExportLogsServiceRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ExportLogsServiceRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.ResourceLogs) > 0 {
		for iNdEx := len(m.ResourceLogs) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.ResourceLogs[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintLogsService(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func (m *ExportLogsServiceResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ExportLogsServiceResponse) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ExportLogsServiceResponse) MarshalToSizedBuffer(dAtA []byte) (int, error) {
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
			i = encodeVarintLogsService(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *ExportLogsPartialSuccess) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *ExportLogsPartialSuccess) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *ExportLogsPartialSuccess) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.ErrorMessage) > 0 {
		i -= len(m.ErrorMessage)
		copy(dAtA[i:], m.ErrorMessage)
		i = encodeVarintLogsService(dAtA, i, uint64(len(m.ErrorMessage)))
		i--
		dAtA[i] = 0x12
	}
	if m.RejectedLogRecords != 0 {
		i = encodeVarintLogsService(dAtA, i, uint64(m.RejectedLogRecords))
		i--
		dAtA[i] = 0x8
	}
	return len(dAtA) - i, nil
}

func encodeVarintLogsService(dAtA []byte, offset int, v uint64) int {
	offset -= sovLogsService(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *ExportLogsServiceRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.ResourceLogs) > 0 {
		for _, e := range m.ResourceLogs {
			l = e.Size()
			n += 1 + l + sovLogsService(uint64(l))
		}
	}
	return n
}

func (m *ExportLogsServiceResponse) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.PartialSuccess != nil {
		l = m.PartialSuccess.Size()
		n += 1 + l + sovLogsService(uint64(l))
	}
	return n
}

func (m *ExportLogsPartialSuccess) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.RejectedLogRecords != 0 {
		n += 1 + sovLogsService(uint64(m.RejectedLogRecords))
	}
	l = len(m.ErrorMessage)
	if l > 0 {
		n += 1 + l + sovLogsService(uint64(l))
	}
	return n
}

func sovLogsService(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozLogsService(x uint64) (n int) {
	return sovLogsService(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *ExportLogsServiceRequest) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForResourceLogs := "[]*ResourceLogs{"
	for _, f := range this.ResourceLogs {
		repeatedStringForResourceLogs += strings.Replace(fmt.Sprintf("%v", f), "ResourceLogs", "v1.ResourceLogs", 1) + ","
	}
	repeatedStringForResourceLogs += "}"
	s := strings.Join([]string{`&ExportLogsServiceRequest{`,
		`ResourceLogs:` + repeatedStringForResourceLogs + `,`,
		`}`,
	}, "")
	return s
}
func (this *ExportLogsServiceResponse) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ExportLogsServiceResponse{`,
		`PartialSuccess:` + strings.Replace(this.PartialSuccess.String(), "ExportLogsPartialSuccess", "ExportLogsPartialSuccess", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *ExportLogsPartialSuccess) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&ExportLogsPartialSuccess{`,
		`RejectedLogRecords:` + fmt.Sprintf("%v", this.RejectedLogRecords) + `,`,
		`ErrorMessage:` + fmt.Sprintf("%v", this.ErrorMessage) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringLogsService(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *ExportLogsServiceRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLogsService
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
			return fmt.Errorf("proto: ExportLogsServiceRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExportLogsServiceRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ResourceLogs", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLogsService
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
				return ErrInvalidLengthLogsService
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthLogsService
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ResourceLogs = append(m.ResourceLogs, &v1.ResourceLogs{})
			if err := m.ResourceLogs[len(m.ResourceLogs)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipLogsService(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthLogsService
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
func (m *ExportLogsServiceResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLogsService
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
			return fmt.Errorf("proto: ExportLogsServiceResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExportLogsServiceResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field PartialSuccess", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLogsService
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
				return ErrInvalidLengthLogsService
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthLogsService
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.PartialSuccess == nil {
				m.PartialSuccess = &ExportLogsPartialSuccess{}
			}
			if err := m.PartialSuccess.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipLogsService(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthLogsService
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
func (m *ExportLogsPartialSuccess) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowLogsService
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
			return fmt.Errorf("proto: ExportLogsPartialSuccess: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: ExportLogsPartialSuccess: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field RejectedLogRecords", wireType)
			}
			m.RejectedLogRecords = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowLogsService
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.RejectedLogRecords |= int64(b&0x7F) << shift
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
					return ErrIntOverflowLogsService
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
				return ErrInvalidLengthLogsService
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthLogsService
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.ErrorMessage = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipLogsService(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthLogsService
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
func skipLogsService(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowLogsService
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
					return 0, ErrIntOverflowLogsService
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
					return 0, ErrIntOverflowLogsService
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
				return 0, ErrInvalidLengthLogsService
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupLogsService
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthLogsService
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthLogsService        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowLogsService          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupLogsService = fmt.Errorf("proto: unexpected end of group")
)
