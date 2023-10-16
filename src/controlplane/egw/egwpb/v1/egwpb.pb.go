// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/controlplane/egw/egwpb/v1/egwpb.proto

package egwpb

import (
	context "context"
	fmt "fmt"
	typespb "gimletlabs.ai/gimlet/src/common/typespb"
	_ "github.com/gogo/protobuf/gogoproto"
	proto "github.com/gogo/protobuf/proto"
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

type RegisterRequest struct {
	DeviceSerial string `protobuf:"bytes,1,opt,name=device_serial,json=deviceSerial,proto3" json:"device_serial,omitempty"`
	Hostname     string `protobuf:"bytes,2,opt,name=hostname,proto3" json:"hostname,omitempty"`
}

func (m *RegisterRequest) Reset()      { *m = RegisterRequest{} }
func (*RegisterRequest) ProtoMessage() {}
func (*RegisterRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_9c5010ad21c5e933, []int{0}
}
func (m *RegisterRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *RegisterRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_RegisterRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *RegisterRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegisterRequest.Merge(m, src)
}
func (m *RegisterRequest) XXX_Size() int {
	return m.Size()
}
func (m *RegisterRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_RegisterRequest.DiscardUnknown(m)
}

var xxx_messageInfo_RegisterRequest proto.InternalMessageInfo

func (m *RegisterRequest) GetDeviceSerial() string {
	if m != nil {
		return m.DeviceSerial
	}
	return ""
}

func (m *RegisterRequest) GetHostname() string {
	if m != nil {
		return m.Hostname
	}
	return ""
}

type RegisterResponse struct {
	Token    string        `protobuf:"bytes,1,opt,name=token,proto3" json:"token,omitempty"`
	DeviceID *typespb.UUID `protobuf:"bytes,2,opt,name=device_id,json=deviceId,proto3" json:"device_id,omitempty"`
}

func (m *RegisterResponse) Reset()      { *m = RegisterResponse{} }
func (*RegisterResponse) ProtoMessage() {}
func (*RegisterResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_9c5010ad21c5e933, []int{1}
}
func (m *RegisterResponse) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *RegisterResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_RegisterResponse.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *RegisterResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_RegisterResponse.Merge(m, src)
}
func (m *RegisterResponse) XXX_Size() int {
	return m.Size()
}
func (m *RegisterResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_RegisterResponse.DiscardUnknown(m)
}

var xxx_messageInfo_RegisterResponse proto.InternalMessageInfo

func (m *RegisterResponse) GetToken() string {
	if m != nil {
		return m.Token
	}
	return ""
}

func (m *RegisterResponse) GetDeviceID() *typespb.UUID {
	if m != nil {
		return m.DeviceID
	}
	return nil
}

type BridgeRequest struct {
}

func (m *BridgeRequest) Reset()      { *m = BridgeRequest{} }
func (*BridgeRequest) ProtoMessage() {}
func (*BridgeRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_9c5010ad21c5e933, []int{2}
}
func (m *BridgeRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *BridgeRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_BridgeRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *BridgeRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BridgeRequest.Merge(m, src)
}
func (m *BridgeRequest) XXX_Size() int {
	return m.Size()
}
func (m *BridgeRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_BridgeRequest.DiscardUnknown(m)
}

var xxx_messageInfo_BridgeRequest proto.InternalMessageInfo

type BridgeResponse struct {
}

func (m *BridgeResponse) Reset()      { *m = BridgeResponse{} }
func (*BridgeResponse) ProtoMessage() {}
func (*BridgeResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_9c5010ad21c5e933, []int{3}
}
func (m *BridgeResponse) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *BridgeResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_BridgeResponse.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *BridgeResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BridgeResponse.Merge(m, src)
}
func (m *BridgeResponse) XXX_Size() int {
	return m.Size()
}
func (m *BridgeResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_BridgeResponse.DiscardUnknown(m)
}

var xxx_messageInfo_BridgeResponse proto.InternalMessageInfo

func init() {
	proto.RegisterType((*RegisterRequest)(nil), "gml.internal.controlplane.egw.v1.RegisterRequest")
	proto.RegisterType((*RegisterResponse)(nil), "gml.internal.controlplane.egw.v1.RegisterResponse")
	proto.RegisterType((*BridgeRequest)(nil), "gml.internal.controlplane.egw.v1.BridgeRequest")
	proto.RegisterType((*BridgeResponse)(nil), "gml.internal.controlplane.egw.v1.BridgeResponse")
}

func init() {
	proto.RegisterFile("src/controlplane/egw/egwpb/v1/egwpb.proto", fileDescriptor_9c5010ad21c5e933)
}

var fileDescriptor_9c5010ad21c5e933 = []byte{
	// 427 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x94, 0x52, 0xcd, 0x6e, 0x13, 0x31,
	0x10, 0x5e, 0x23, 0x51, 0xa5, 0xa6, 0x25, 0x95, 0xc5, 0xa1, 0x5a, 0x09, 0x53, 0x85, 0x4b, 0xb9,
	0xd8, 0x4d, 0xb8, 0xf0, 0x73, 0x8b, 0x82, 0x50, 0xae, 0x5b, 0x55, 0x48, 0x5c, 0xaa, 0xdd, 0xec,
	0xe0, 0x58, 0xec, 0xda, 0x5b, 0xdb, 0x9b, 0xaa, 0x37, 0x1e, 0x81, 0xc7, 0xe0, 0x51, 0x38, 0xe6,
	0xd8, 0x13, 0x22, 0xce, 0x85, 0x63, 0x1e, 0x01, 0xc5, 0xde, 0xf2, 0x77, 0xa0, 0x70, 0xb0, 0x34,
	0xdf, 0x78, 0xbe, 0x6f, 0xbe, 0x19, 0x0d, 0x7e, 0x62, 0xcd, 0x8c, 0xcf, 0xb4, 0x72, 0x46, 0x57,
	0x4d, 0x95, 0x2b, 0xe0, 0x20, 0x2e, 0xb7, 0xaf, 0x29, 0xf8, 0x62, 0x18, 0x03, 0xd6, 0x18, 0xed,
	0x34, 0x39, 0x12, 0x75, 0xc5, 0xa4, 0x72, 0x60, 0x54, 0x5e, 0xb1, 0x5f, 0x39, 0x0c, 0xc4, 0x25,
	0x5b, 0x0c, 0xd3, 0xe7, 0x6e, 0x2e, 0x4d, 0x79, 0xde, 0xe4, 0xc6, 0x5d, 0x71, 0x21, 0xdd, 0xbc,
	0x2d, 0xd8, 0x4c, 0xd7, 0x5c, 0x68, 0xa1, 0x79, 0x10, 0x29, 0xda, 0x77, 0x01, 0x05, 0x10, 0xa2,
	0x28, 0x9e, 0x3e, 0x8c, 0x3e, 0xea, 0x5a, 0x2b, 0xee, 0xae, 0x1a, 0xb0, 0x4d, 0xc1, 0xdb, 0x56,
	0x96, 0xf1, 0x7b, 0x90, 0xe1, 0x7e, 0x06, 0x42, 0x5a, 0x07, 0x26, 0x83, 0x8b, 0x16, 0xac, 0x23,
	0x8f, 0xf1, 0x7e, 0x09, 0x0b, 0x39, 0x83, 0x73, 0x0b, 0x46, 0xe6, 0xd5, 0x21, 0x3a, 0x42, 0xc7,
	0xbb, 0xd9, 0x5e, 0x4c, 0x9e, 0x86, 0x1c, 0x49, 0x71, 0x6f, 0xae, 0xad, 0x53, 0x79, 0x0d, 0x87,
	0x77, 0xc2, 0xff, 0x0f, 0x3c, 0x28, 0xf1, 0xc1, 0x4f, 0x4d, 0xdb, 0x68, 0x65, 0x81, 0x3c, 0xc0,
	0x77, 0x9d, 0x7e, 0x0f, 0xaa, 0x13, 0x8b, 0x80, 0xbc, 0xc0, 0xbb, 0x5d, 0x2b, 0x59, 0x06, 0x99,
	0x7b, 0xa3, 0x3e, 0xdb, 0x6e, 0x23, 0x38, 0x65, 0x67, 0x67, 0xd3, 0xc9, 0x78, 0xcf, 0x7f, 0x79,
	0xd4, 0x9b, 0x84, 0xaa, 0xe9, 0x24, 0xeb, 0xc5, 0xfa, 0x69, 0x39, 0xe8, 0xe3, 0xfd, 0xb1, 0x91,
	0xa5, 0x80, 0xce, 0xf7, 0xe0, 0x00, 0xdf, 0xbf, 0x49, 0xc4, 0xa6, 0xa3, 0x0d, 0xc2, 0xf8, 0xd5,
	0xeb, 0x37, 0xa7, 0x60, 0xb6, 0x1c, 0x72, 0x81, 0x7b, 0x37, 0xbe, 0xc8, 0x90, 0xdd, 0xb6, 0x74,
	0xf6, 0xc7, 0x5e, 0xd2, 0xd1, 0xff, 0x50, 0xba, 0xb1, 0x35, 0xde, 0x89, 0x9e, 0x08, 0xbf, 0x9d,
	0xfd, 0xdb, 0x38, 0xe9, 0xc9, 0xbf, 0x13, 0x62, 0xb3, 0x63, 0x74, 0x82, 0xc6, 0x6a, 0xb9, 0xa2,
	0xc9, 0xf5, 0x8a, 0x26, 0x9b, 0x15, 0x45, 0x1f, 0x3c, 0x45, 0x9f, 0x3c, 0x45, 0x9f, 0x3d, 0x45,
	0x4b, 0x4f, 0xd1, 0x57, 0x4f, 0xd1, 0x37, 0x4f, 0x93, 0x8d, 0xa7, 0xe8, 0xe3, 0x9a, 0x26, 0xcb,
	0x35, 0x4d, 0xae, 0xd7, 0x34, 0x79, 0xfb, 0x4c, 0xc8, 0xba, 0x02, 0x57, 0xe5, 0x85, 0x65, 0xb9,
	0xe4, 0x11, 0xf1, 0xbf, 0x5e, 0xf0, 0xcb, 0x10, 0x14, 0x3b, 0xe1, 0x8c, 0x9e, 0x7e, 0x0f, 0x00,
	0x00, 0xff, 0xff, 0x5a, 0x06, 0xa5, 0xdc, 0xef, 0x02, 0x00, 0x00,
}

func (this *RegisterRequest) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*RegisterRequest)
	if !ok {
		that2, ok := that.(RegisterRequest)
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
	if this.DeviceSerial != that1.DeviceSerial {
		return false
	}
	if this.Hostname != that1.Hostname {
		return false
	}
	return true
}
func (this *RegisterResponse) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*RegisterResponse)
	if !ok {
		that2, ok := that.(RegisterResponse)
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
	if this.Token != that1.Token {
		return false
	}
	if !this.DeviceID.Equal(that1.DeviceID) {
		return false
	}
	return true
}
func (this *BridgeRequest) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*BridgeRequest)
	if !ok {
		that2, ok := that.(BridgeRequest)
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
func (this *BridgeResponse) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*BridgeResponse)
	if !ok {
		that2, ok := that.(BridgeResponse)
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
func (this *RegisterRequest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&egwpb.RegisterRequest{")
	s = append(s, "DeviceSerial: "+fmt.Sprintf("%#v", this.DeviceSerial)+",\n")
	s = append(s, "Hostname: "+fmt.Sprintf("%#v", this.Hostname)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *RegisterResponse) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&egwpb.RegisterResponse{")
	s = append(s, "Token: "+fmt.Sprintf("%#v", this.Token)+",\n")
	if this.DeviceID != nil {
		s = append(s, "DeviceID: "+fmt.Sprintf("%#v", this.DeviceID)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *BridgeRequest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 4)
	s = append(s, "&egwpb.BridgeRequest{")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *BridgeResponse) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 4)
	s = append(s, "&egwpb.BridgeResponse{")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringEgwpb(v interface{}, typ string) string {
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

// EGWServiceClient is the client API for EGWService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type EGWServiceClient interface {
	Register(ctx context.Context, in *RegisterRequest, opts ...grpc.CallOption) (*RegisterResponse, error)
	Bridge(ctx context.Context, opts ...grpc.CallOption) (EGWService_BridgeClient, error)
}

type eGWServiceClient struct {
	cc *grpc.ClientConn
}

func NewEGWServiceClient(cc *grpc.ClientConn) EGWServiceClient {
	return &eGWServiceClient{cc}
}

func (c *eGWServiceClient) Register(ctx context.Context, in *RegisterRequest, opts ...grpc.CallOption) (*RegisterResponse, error) {
	out := new(RegisterResponse)
	err := c.cc.Invoke(ctx, "/gml.internal.controlplane.egw.v1.EGWService/Register", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *eGWServiceClient) Bridge(ctx context.Context, opts ...grpc.CallOption) (EGWService_BridgeClient, error) {
	stream, err := c.cc.NewStream(ctx, &_EGWService_serviceDesc.Streams[0], "/gml.internal.controlplane.egw.v1.EGWService/Bridge", opts...)
	if err != nil {
		return nil, err
	}
	x := &eGWServiceBridgeClient{stream}
	return x, nil
}

type EGWService_BridgeClient interface {
	Send(*BridgeRequest) error
	Recv() (*BridgeResponse, error)
	grpc.ClientStream
}

type eGWServiceBridgeClient struct {
	grpc.ClientStream
}

func (x *eGWServiceBridgeClient) Send(m *BridgeRequest) error {
	return x.ClientStream.SendMsg(m)
}

func (x *eGWServiceBridgeClient) Recv() (*BridgeResponse, error) {
	m := new(BridgeResponse)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// EGWServiceServer is the server API for EGWService service.
type EGWServiceServer interface {
	Register(context.Context, *RegisterRequest) (*RegisterResponse, error)
	Bridge(EGWService_BridgeServer) error
}

// UnimplementedEGWServiceServer can be embedded to have forward compatible implementations.
type UnimplementedEGWServiceServer struct {
}

func (*UnimplementedEGWServiceServer) Register(ctx context.Context, req *RegisterRequest) (*RegisterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Register not implemented")
}
func (*UnimplementedEGWServiceServer) Bridge(srv EGWService_BridgeServer) error {
	return status.Errorf(codes.Unimplemented, "method Bridge not implemented")
}

func RegisterEGWServiceServer(s *grpc.Server, srv EGWServiceServer) {
	s.RegisterService(&_EGWService_serviceDesc, srv)
}

func _EGWService_Register_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RegisterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EGWServiceServer).Register(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/gml.internal.controlplane.egw.v1.EGWService/Register",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EGWServiceServer).Register(ctx, req.(*RegisterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EGWService_Bridge_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(EGWServiceServer).Bridge(&eGWServiceBridgeServer{stream})
}

type EGWService_BridgeServer interface {
	Send(*BridgeResponse) error
	Recv() (*BridgeRequest, error)
	grpc.ServerStream
}

type eGWServiceBridgeServer struct {
	grpc.ServerStream
}

func (x *eGWServiceBridgeServer) Send(m *BridgeResponse) error {
	return x.ServerStream.SendMsg(m)
}

func (x *eGWServiceBridgeServer) Recv() (*BridgeRequest, error) {
	m := new(BridgeRequest)
	if err := x.ServerStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

var _EGWService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "gml.internal.controlplane.egw.v1.EGWService",
	HandlerType: (*EGWServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Register",
			Handler:    _EGWService_Register_Handler,
		},
	},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "Bridge",
			Handler:       _EGWService_Bridge_Handler,
			ServerStreams: true,
			ClientStreams: true,
		},
	},
	Metadata: "src/controlplane/egw/egwpb/v1/egwpb.proto",
}

func (m *RegisterRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *RegisterRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *RegisterRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Hostname) > 0 {
		i -= len(m.Hostname)
		copy(dAtA[i:], m.Hostname)
		i = encodeVarintEgwpb(dAtA, i, uint64(len(m.Hostname)))
		i--
		dAtA[i] = 0x12
	}
	if len(m.DeviceSerial) > 0 {
		i -= len(m.DeviceSerial)
		copy(dAtA[i:], m.DeviceSerial)
		i = encodeVarintEgwpb(dAtA, i, uint64(len(m.DeviceSerial)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *RegisterResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *RegisterResponse) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *RegisterResponse) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.DeviceID != nil {
		{
			size, err := m.DeviceID.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintEgwpb(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	if len(m.Token) > 0 {
		i -= len(m.Token)
		copy(dAtA[i:], m.Token)
		i = encodeVarintEgwpb(dAtA, i, uint64(len(m.Token)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *BridgeRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *BridgeRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *BridgeRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	return len(dAtA) - i, nil
}

func (m *BridgeResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *BridgeResponse) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *BridgeResponse) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	return len(dAtA) - i, nil
}

func encodeVarintEgwpb(dAtA []byte, offset int, v uint64) int {
	offset -= sovEgwpb(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *RegisterRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.DeviceSerial)
	if l > 0 {
		n += 1 + l + sovEgwpb(uint64(l))
	}
	l = len(m.Hostname)
	if l > 0 {
		n += 1 + l + sovEgwpb(uint64(l))
	}
	return n
}

func (m *RegisterResponse) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Token)
	if l > 0 {
		n += 1 + l + sovEgwpb(uint64(l))
	}
	if m.DeviceID != nil {
		l = m.DeviceID.Size()
		n += 1 + l + sovEgwpb(uint64(l))
	}
	return n
}

func (m *BridgeRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	return n
}

func (m *BridgeResponse) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	return n
}

func sovEgwpb(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozEgwpb(x uint64) (n int) {
	return sovEgwpb(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *RegisterRequest) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&RegisterRequest{`,
		`DeviceSerial:` + fmt.Sprintf("%v", this.DeviceSerial) + `,`,
		`Hostname:` + fmt.Sprintf("%v", this.Hostname) + `,`,
		`}`,
	}, "")
	return s
}
func (this *RegisterResponse) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&RegisterResponse{`,
		`Token:` + fmt.Sprintf("%v", this.Token) + `,`,
		`DeviceID:` + strings.Replace(fmt.Sprintf("%v", this.DeviceID), "UUID", "typespb.UUID", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *BridgeRequest) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&BridgeRequest{`,
		`}`,
	}, "")
	return s
}
func (this *BridgeResponse) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&BridgeResponse{`,
		`}`,
	}, "")
	return s
}
func valueToStringEgwpb(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *RegisterRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowEgwpb
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
			return fmt.Errorf("proto: RegisterRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: RegisterRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field DeviceSerial", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowEgwpb
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
				return ErrInvalidLengthEgwpb
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthEgwpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.DeviceSerial = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Hostname", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowEgwpb
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
				return ErrInvalidLengthEgwpb
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthEgwpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Hostname = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipEgwpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthEgwpb
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
func (m *RegisterResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowEgwpb
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
			return fmt.Errorf("proto: RegisterResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: RegisterResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Token", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowEgwpb
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
				return ErrInvalidLengthEgwpb
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthEgwpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Token = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field DeviceID", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowEgwpb
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
				return ErrInvalidLengthEgwpb
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthEgwpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.DeviceID == nil {
				m.DeviceID = &typespb.UUID{}
			}
			if err := m.DeviceID.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipEgwpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthEgwpb
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
func (m *BridgeRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowEgwpb
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
			return fmt.Errorf("proto: BridgeRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: BridgeRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		default:
			iNdEx = preIndex
			skippy, err := skipEgwpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthEgwpb
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
func (m *BridgeResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowEgwpb
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
			return fmt.Errorf("proto: BridgeResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: BridgeResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		default:
			iNdEx = preIndex
			skippy, err := skipEgwpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthEgwpb
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
func skipEgwpb(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowEgwpb
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
					return 0, ErrIntOverflowEgwpb
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
					return 0, ErrIntOverflowEgwpb
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
				return 0, ErrInvalidLengthEgwpb
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupEgwpb
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthEgwpb
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthEgwpb        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowEgwpb          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupEgwpb = fmt.Errorf("proto: unexpected end of group")
)
