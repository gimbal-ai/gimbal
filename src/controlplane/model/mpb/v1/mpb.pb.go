// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: src/controlplane/model/mpb/v1/mpb.proto

package mpb

import (
	context "context"
	fmt "fmt"
	v1 "gimletlabs.ai/gimlet/src/api/corepb/v1"
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

type GetModelRequest struct {
	ID    *typespb.UUID `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`
	Name  string        `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
	OrgID *typespb.UUID `protobuf:"bytes,3,opt,name=org_id,json=orgId,proto3" json:"org_id,omitempty"`
}

func (m *GetModelRequest) Reset()      { *m = GetModelRequest{} }
func (*GetModelRequest) ProtoMessage() {}
func (*GetModelRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_6328f2b6803fd88b, []int{0}
}
func (m *GetModelRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *GetModelRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_GetModelRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *GetModelRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GetModelRequest.Merge(m, src)
}
func (m *GetModelRequest) XXX_Size() int {
	return m.Size()
}
func (m *GetModelRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_GetModelRequest.DiscardUnknown(m)
}

var xxx_messageInfo_GetModelRequest proto.InternalMessageInfo

func (m *GetModelRequest) GetID() *typespb.UUID {
	if m != nil {
		return m.ID
	}
	return nil
}

func (m *GetModelRequest) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *GetModelRequest) GetOrgID() *typespb.UUID {
	if m != nil {
		return m.OrgID
	}
	return nil
}

type GetModelResponse struct {
	ModelInfo *v1.ModelInfo `protobuf:"bytes,1,opt,name=model_info,json=modelInfo,proto3" json:"model_info,omitempty"`
}

func (m *GetModelResponse) Reset()      { *m = GetModelResponse{} }
func (*GetModelResponse) ProtoMessage() {}
func (*GetModelResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_6328f2b6803fd88b, []int{1}
}
func (m *GetModelResponse) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *GetModelResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_GetModelResponse.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *GetModelResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_GetModelResponse.Merge(m, src)
}
func (m *GetModelResponse) XXX_Size() int {
	return m.Size()
}
func (m *GetModelResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_GetModelResponse.DiscardUnknown(m)
}

var xxx_messageInfo_GetModelResponse proto.InternalMessageInfo

func (m *GetModelResponse) GetModelInfo() *v1.ModelInfo {
	if m != nil {
		return m.ModelInfo
	}
	return nil
}

type CreateModelRequest struct {
	OrgID     *typespb.UUID `protobuf:"bytes,1,opt,name=org_id,json=orgId,proto3" json:"org_id,omitempty"`
	Name      string        `protobuf:"bytes,2,opt,name=name,proto3" json:"name,omitempty"`
	ModelInfo *v1.ModelInfo `protobuf:"bytes,3,opt,name=model_info,json=modelInfo,proto3" json:"model_info,omitempty"`
}

func (m *CreateModelRequest) Reset()      { *m = CreateModelRequest{} }
func (*CreateModelRequest) ProtoMessage() {}
func (*CreateModelRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_6328f2b6803fd88b, []int{2}
}
func (m *CreateModelRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *CreateModelRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_CreateModelRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *CreateModelRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CreateModelRequest.Merge(m, src)
}
func (m *CreateModelRequest) XXX_Size() int {
	return m.Size()
}
func (m *CreateModelRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_CreateModelRequest.DiscardUnknown(m)
}

var xxx_messageInfo_CreateModelRequest proto.InternalMessageInfo

func (m *CreateModelRequest) GetOrgID() *typespb.UUID {
	if m != nil {
		return m.OrgID
	}
	return nil
}

func (m *CreateModelRequest) GetName() string {
	if m != nil {
		return m.Name
	}
	return ""
}

func (m *CreateModelRequest) GetModelInfo() *v1.ModelInfo {
	if m != nil {
		return m.ModelInfo
	}
	return nil
}

type CreateModelResponse struct {
	ID *typespb.UUID `protobuf:"bytes,1,opt,name=id,proto3" json:"id,omitempty"`
}

func (m *CreateModelResponse) Reset()      { *m = CreateModelResponse{} }
func (*CreateModelResponse) ProtoMessage() {}
func (*CreateModelResponse) Descriptor() ([]byte, []int) {
	return fileDescriptor_6328f2b6803fd88b, []int{3}
}
func (m *CreateModelResponse) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *CreateModelResponse) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_CreateModelResponse.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *CreateModelResponse) XXX_Merge(src proto.Message) {
	xxx_messageInfo_CreateModelResponse.Merge(m, src)
}
func (m *CreateModelResponse) XXX_Size() int {
	return m.Size()
}
func (m *CreateModelResponse) XXX_DiscardUnknown() {
	xxx_messageInfo_CreateModelResponse.DiscardUnknown(m)
}

var xxx_messageInfo_CreateModelResponse proto.InternalMessageInfo

func (m *CreateModelResponse) GetID() *typespb.UUID {
	if m != nil {
		return m.ID
	}
	return nil
}

func init() {
	proto.RegisterType((*GetModelRequest)(nil), "gml.internal.controlplane.model.v1.GetModelRequest")
	proto.RegisterType((*GetModelResponse)(nil), "gml.internal.controlplane.model.v1.GetModelResponse")
	proto.RegisterType((*CreateModelRequest)(nil), "gml.internal.controlplane.model.v1.CreateModelRequest")
	proto.RegisterType((*CreateModelResponse)(nil), "gml.internal.controlplane.model.v1.CreateModelResponse")
}

func init() {
	proto.RegisterFile("src/controlplane/model/mpb/v1/mpb.proto", fileDescriptor_6328f2b6803fd88b)
}

var fileDescriptor_6328f2b6803fd88b = []byte{
	// 465 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x9c, 0x93, 0x3d, 0x6b, 0x14, 0x41,
	0x18, 0xc7, 0x77, 0x36, 0x26, 0x78, 0x13, 0x21, 0x32, 0x5a, 0x1c, 0x07, 0x4e, 0xc2, 0x5a, 0x24,
	0xd5, 0x0c, 0x97, 0x48, 0x2c, 0x04, 0x8b, 0xf3, 0x40, 0xb6, 0x10, 0x61, 0x25, 0x16, 0x36, 0x61,
	0x5f, 0x9e, 0x2c, 0x03, 0x3b, 0x2f, 0xce, 0xee, 0x1d, 0xda, 0x88, 0x16, 0xf6, 0x7e, 0x03, 0x5b,
	0x3f, 0x8a, 0xe5, 0x95, 0xa9, 0x82, 0x37, 0xd7, 0x58, 0xe6, 0x23, 0xc8, 0xbe, 0x84, 0xe4, 0x34,
	0xe8, 0xc5, 0x6a, 0x9f, 0xdd, 0x99, 0xff, 0xff, 0xf9, 0xfd, 0xe7, 0xd9, 0xc1, 0xbb, 0xa5, 0x4d,
	0x79, 0xaa, 0x55, 0x65, 0x75, 0x61, 0x8a, 0x58, 0x01, 0x97, 0x3a, 0x83, 0x82, 0x4b, 0x93, 0xf0,
	0xe9, 0xb0, 0x7e, 0x30, 0x63, 0x75, 0xa5, 0x49, 0x90, 0xcb, 0x82, 0x09, 0x55, 0x81, 0x55, 0x71,
	0xc1, 0xae, 0x2a, 0x58, 0xa3, 0x60, 0xd3, 0xe1, 0xe0, 0x7e, 0xae, 0x73, 0xdd, 0x6c, 0xe7, 0x75,
	0xd5, 0x2a, 0x07, 0x0f, 0xda, 0x16, 0x52, 0x6a, 0xc5, 0xab, 0xf7, 0x06, 0x4a, 0x93, 0xf0, 0xc9,
	0x44, 0x64, 0xdd, 0x72, 0x50, 0x2f, 0xc7, 0x46, 0xf0, 0x54, 0x5b, 0xe8, 0xba, 0xd6, 0x86, 0xc7,
	0xf0, 0x0e, 0xd2, 0x76, 0x4f, 0xf0, 0x09, 0xe1, 0xad, 0xe7, 0x50, 0xbd, 0xa8, 0xbf, 0x47, 0xf0,
	0x76, 0x02, 0x65, 0x45, 0x76, 0xb1, 0x2f, 0xb2, 0x3e, 0xda, 0x41, 0x7b, 0x9b, 0xfb, 0x5b, 0xac,
	0xa6, 0x6b, 0xcc, 0xd9, 0xd1, 0x51, 0x38, 0x1e, 0x6d, 0xb8, 0xb3, 0x6d, 0x3f, 0x1c, 0x47, 0xbe,
	0xc8, 0x08, 0xc1, 0xb7, 0x54, 0x2c, 0xa1, 0xef, 0xef, 0xa0, 0xbd, 0x5e, 0xd4, 0xd4, 0x64, 0x88,
	0x37, 0xb4, 0xcd, 0x8f, 0x45, 0xd6, 0x5f, 0xbb, 0xde, 0xa0, 0xe7, 0xce, 0xb6, 0xd7, 0x5f, 0xda,
	0x3c, 0x1c, 0x47, 0xeb, 0xda, 0xe6, 0x61, 0x16, 0xbc, 0xc6, 0x77, 0x2f, 0x11, 0x4a, 0xa3, 0x55,
	0x09, 0x64, 0x84, 0x71, 0xcb, 0x2a, 0xd4, 0x89, 0xee, 0x58, 0x1e, 0xb2, 0xa5, 0x93, 0x8a, 0x8d,
	0x60, 0x75, 0x32, 0x36, 0x1d, 0xb2, 0x46, 0x1c, 0xaa, 0x13, 0x1d, 0xf5, 0xe4, 0x45, 0x19, 0x7c,
	0x45, 0x98, 0x3c, 0xb3, 0x10, 0x57, 0xb0, 0x14, 0xef, 0x92, 0x10, 0xad, 0x48, 0x78, 0x6d, 0xd0,
	0x65, 0xc2, 0xb5, 0xff, 0x22, 0x7c, 0x8a, 0xef, 0x2d, 0x01, 0x76, 0xe1, 0x57, 0x1d, 0xc0, 0xfe,
	0x67, 0x1f, 0xdf, 0x69, 0xa4, 0xaf, 0xc0, 0x4e, 0x45, 0x0a, 0x64, 0x82, 0x6f, 0x5f, 0x1c, 0x25,
	0x39, 0x60, 0xff, 0xfe, 0xb1, 0xd8, 0x6f, 0xb3, 0x1f, 0x3c, 0xba, 0x99, 0xa8, 0x03, 0xfe, 0x80,
	0x37, 0xaf, 0xe4, 0x20, 0x87, 0xab, 0x98, 0xfc, 0x39, 0x99, 0xc1, 0xe3, 0x1b, 0xeb, 0xda, 0xfe,
	0xa3, 0x62, 0x36, 0xa7, 0xde, 0xe9, 0x9c, 0x7a, 0xe7, 0x73, 0x8a, 0x3e, 0x3a, 0x8a, 0xbe, 0x39,
	0x8a, 0xbe, 0x3b, 0x8a, 0x66, 0x8e, 0xa2, 0x1f, 0x8e, 0xa2, 0x9f, 0x8e, 0x7a, 0xe7, 0x8e, 0xa2,
	0x2f, 0x0b, 0xea, 0xcd, 0x16, 0xd4, 0x3b, 0x5d, 0x50, 0xef, 0xcd, 0x61, 0x2e, 0x64, 0x01, 0x55,
	0x11, 0x27, 0x25, 0x8b, 0x05, 0x6f, 0xdf, 0xf8, 0x5f, 0xaf, 0xed, 0x13, 0x69, 0x92, 0x64, 0xa3,
	0xb9, 0x3a, 0x07, 0xbf, 0x02, 0x00, 0x00, 0xff, 0xff, 0x47, 0x85, 0x9f, 0xe4, 0xe2, 0x03, 0x00,
	0x00,
}

func (this *GetModelRequest) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*GetModelRequest)
	if !ok {
		that2, ok := that.(GetModelRequest)
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
	if !this.ID.Equal(that1.ID) {
		return false
	}
	if this.Name != that1.Name {
		return false
	}
	if !this.OrgID.Equal(that1.OrgID) {
		return false
	}
	return true
}
func (this *GetModelResponse) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*GetModelResponse)
	if !ok {
		that2, ok := that.(GetModelResponse)
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
	if !this.ModelInfo.Equal(that1.ModelInfo) {
		return false
	}
	return true
}
func (this *CreateModelRequest) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*CreateModelRequest)
	if !ok {
		that2, ok := that.(CreateModelRequest)
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
	if !this.OrgID.Equal(that1.OrgID) {
		return false
	}
	if this.Name != that1.Name {
		return false
	}
	if !this.ModelInfo.Equal(that1.ModelInfo) {
		return false
	}
	return true
}
func (this *CreateModelResponse) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*CreateModelResponse)
	if !ok {
		that2, ok := that.(CreateModelResponse)
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
	if !this.ID.Equal(that1.ID) {
		return false
	}
	return true
}
func (this *GetModelRequest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&mpb.GetModelRequest{")
	if this.ID != nil {
		s = append(s, "ID: "+fmt.Sprintf("%#v", this.ID)+",\n")
	}
	s = append(s, "Name: "+fmt.Sprintf("%#v", this.Name)+",\n")
	if this.OrgID != nil {
		s = append(s, "OrgID: "+fmt.Sprintf("%#v", this.OrgID)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *GetModelResponse) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&mpb.GetModelResponse{")
	if this.ModelInfo != nil {
		s = append(s, "ModelInfo: "+fmt.Sprintf("%#v", this.ModelInfo)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *CreateModelRequest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&mpb.CreateModelRequest{")
	if this.OrgID != nil {
		s = append(s, "OrgID: "+fmt.Sprintf("%#v", this.OrgID)+",\n")
	}
	s = append(s, "Name: "+fmt.Sprintf("%#v", this.Name)+",\n")
	if this.ModelInfo != nil {
		s = append(s, "ModelInfo: "+fmt.Sprintf("%#v", this.ModelInfo)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *CreateModelResponse) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&mpb.CreateModelResponse{")
	if this.ID != nil {
		s = append(s, "ID: "+fmt.Sprintf("%#v", this.ID)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringMpb(v interface{}, typ string) string {
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

// ModelServiceClient is the client API for ModelService service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type ModelServiceClient interface {
	GetModel(ctx context.Context, in *GetModelRequest, opts ...grpc.CallOption) (*GetModelResponse, error)
	CreateModel(ctx context.Context, in *CreateModelRequest, opts ...grpc.CallOption) (*CreateModelResponse, error)
}

type modelServiceClient struct {
	cc *grpc.ClientConn
}

func NewModelServiceClient(cc *grpc.ClientConn) ModelServiceClient {
	return &modelServiceClient{cc}
}

func (c *modelServiceClient) GetModel(ctx context.Context, in *GetModelRequest, opts ...grpc.CallOption) (*GetModelResponse, error) {
	out := new(GetModelResponse)
	err := c.cc.Invoke(ctx, "/gml.internal.controlplane.model.v1.ModelService/GetModel", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelServiceClient) CreateModel(ctx context.Context, in *CreateModelRequest, opts ...grpc.CallOption) (*CreateModelResponse, error) {
	out := new(CreateModelResponse)
	err := c.cc.Invoke(ctx, "/gml.internal.controlplane.model.v1.ModelService/CreateModel", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ModelServiceServer is the server API for ModelService service.
type ModelServiceServer interface {
	GetModel(context.Context, *GetModelRequest) (*GetModelResponse, error)
	CreateModel(context.Context, *CreateModelRequest) (*CreateModelResponse, error)
}

// UnimplementedModelServiceServer can be embedded to have forward compatible implementations.
type UnimplementedModelServiceServer struct {
}

func (*UnimplementedModelServiceServer) GetModel(ctx context.Context, req *GetModelRequest) (*GetModelResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetModel not implemented")
}
func (*UnimplementedModelServiceServer) CreateModel(ctx context.Context, req *CreateModelRequest) (*CreateModelResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method CreateModel not implemented")
}

func RegisterModelServiceServer(s *grpc.Server, srv ModelServiceServer) {
	s.RegisterService(&_ModelService_serviceDesc, srv)
}

func _ModelService_GetModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetModelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).GetModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/gml.internal.controlplane.model.v1.ModelService/GetModel",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).GetModel(ctx, req.(*GetModelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ModelService_CreateModel_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CreateModelRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServiceServer).CreateModel(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/gml.internal.controlplane.model.v1.ModelService/CreateModel",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServiceServer).CreateModel(ctx, req.(*CreateModelRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _ModelService_serviceDesc = grpc.ServiceDesc{
	ServiceName: "gml.internal.controlplane.model.v1.ModelService",
	HandlerType: (*ModelServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "GetModel",
			Handler:    _ModelService_GetModel_Handler,
		},
		{
			MethodName: "CreateModel",
			Handler:    _ModelService_CreateModel_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "src/controlplane/model/mpb/v1/mpb.proto",
}

func (m *GetModelRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *GetModelRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *GetModelRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.OrgID != nil {
		{
			size, err := m.OrgID.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintMpb(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x1a
	}
	if len(m.Name) > 0 {
		i -= len(m.Name)
		copy(dAtA[i:], m.Name)
		i = encodeVarintMpb(dAtA, i, uint64(len(m.Name)))
		i--
		dAtA[i] = 0x12
	}
	if m.ID != nil {
		{
			size, err := m.ID.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintMpb(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *GetModelResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *GetModelResponse) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *GetModelResponse) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.ModelInfo != nil {
		{
			size, err := m.ModelInfo.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintMpb(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *CreateModelRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *CreateModelRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *CreateModelRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.ModelInfo != nil {
		{
			size, err := m.ModelInfo.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintMpb(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x1a
	}
	if len(m.Name) > 0 {
		i -= len(m.Name)
		copy(dAtA[i:], m.Name)
		i = encodeVarintMpb(dAtA, i, uint64(len(m.Name)))
		i--
		dAtA[i] = 0x12
	}
	if m.OrgID != nil {
		{
			size, err := m.OrgID.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintMpb(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func (m *CreateModelResponse) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *CreateModelResponse) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *CreateModelResponse) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.ID != nil {
		{
			size, err := m.ID.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintMpb(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintMpb(dAtA []byte, offset int, v uint64) int {
	offset -= sovMpb(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *GetModelRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.ID != nil {
		l = m.ID.Size()
		n += 1 + l + sovMpb(uint64(l))
	}
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovMpb(uint64(l))
	}
	if m.OrgID != nil {
		l = m.OrgID.Size()
		n += 1 + l + sovMpb(uint64(l))
	}
	return n
}

func (m *GetModelResponse) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.ModelInfo != nil {
		l = m.ModelInfo.Size()
		n += 1 + l + sovMpb(uint64(l))
	}
	return n
}

func (m *CreateModelRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.OrgID != nil {
		l = m.OrgID.Size()
		n += 1 + l + sovMpb(uint64(l))
	}
	l = len(m.Name)
	if l > 0 {
		n += 1 + l + sovMpb(uint64(l))
	}
	if m.ModelInfo != nil {
		l = m.ModelInfo.Size()
		n += 1 + l + sovMpb(uint64(l))
	}
	return n
}

func (m *CreateModelResponse) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.ID != nil {
		l = m.ID.Size()
		n += 1 + l + sovMpb(uint64(l))
	}
	return n
}

func sovMpb(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozMpb(x uint64) (n int) {
	return sovMpb(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *GetModelRequest) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&GetModelRequest{`,
		`ID:` + strings.Replace(fmt.Sprintf("%v", this.ID), "UUID", "typespb.UUID", 1) + `,`,
		`Name:` + fmt.Sprintf("%v", this.Name) + `,`,
		`OrgID:` + strings.Replace(fmt.Sprintf("%v", this.OrgID), "UUID", "typespb.UUID", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *GetModelResponse) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&GetModelResponse{`,
		`ModelInfo:` + strings.Replace(fmt.Sprintf("%v", this.ModelInfo), "ModelInfo", "v1.ModelInfo", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *CreateModelRequest) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&CreateModelRequest{`,
		`OrgID:` + strings.Replace(fmt.Sprintf("%v", this.OrgID), "UUID", "typespb.UUID", 1) + `,`,
		`Name:` + fmt.Sprintf("%v", this.Name) + `,`,
		`ModelInfo:` + strings.Replace(fmt.Sprintf("%v", this.ModelInfo), "ModelInfo", "v1.ModelInfo", 1) + `,`,
		`}`,
	}, "")
	return s
}
func (this *CreateModelResponse) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&CreateModelResponse{`,
		`ID:` + strings.Replace(fmt.Sprintf("%v", this.ID), "UUID", "typespb.UUID", 1) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringMpb(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *GetModelRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMpb
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
			return fmt.Errorf("proto: GetModelRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: GetModelRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ID", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ID == nil {
				m.ID = &typespb.UUID{}
			}
			if err := m.ID.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OrgID", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.OrgID == nil {
				m.OrgID = &typespb.UUID{}
			}
			if err := m.OrgID.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipMpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMpb
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
func (m *GetModelResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMpb
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
			return fmt.Errorf("proto: GetModelResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: GetModelResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelInfo", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ModelInfo == nil {
				m.ModelInfo = &v1.ModelInfo{}
			}
			if err := m.ModelInfo.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipMpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMpb
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
func (m *CreateModelRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMpb
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
			return fmt.Errorf("proto: CreateModelRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: CreateModelRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field OrgID", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.OrgID == nil {
				m.OrgID = &typespb.UUID{}
			}
			if err := m.OrgID.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Name", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Name = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ModelInfo", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ModelInfo == nil {
				m.ModelInfo = &v1.ModelInfo{}
			}
			if err := m.ModelInfo.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipMpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMpb
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
func (m *CreateModelResponse) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowMpb
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
			return fmt.Errorf("proto: CreateModelResponse: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: CreateModelResponse: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field ID", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowMpb
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
				return ErrInvalidLengthMpb
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthMpb
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.ID == nil {
				m.ID = &typespb.UUID{}
			}
			if err := m.ID.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipMpb(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthMpb
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
func skipMpb(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowMpb
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
					return 0, ErrIntOverflowMpb
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
					return 0, ErrIntOverflowMpb
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
				return 0, ErrInvalidLengthMpb
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupMpb
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthMpb
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthMpb        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowMpb          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupMpb = fmt.Errorf("proto: unexpected end of group")
)