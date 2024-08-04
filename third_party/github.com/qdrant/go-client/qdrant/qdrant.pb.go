// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: qdrant/qdrant.proto

package qdrant

import (
	context "context"
	fmt "fmt"
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

type HealthCheckRequest struct {
}

func (m *HealthCheckRequest) Reset()      { *m = HealthCheckRequest{} }
func (*HealthCheckRequest) ProtoMessage() {}
func (*HealthCheckRequest) Descriptor() ([]byte, []int) {
	return fileDescriptor_e0e9c33bd10e4006, []int{0}
}
func (m *HealthCheckRequest) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *HealthCheckRequest) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_HealthCheckRequest.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *HealthCheckRequest) XXX_Merge(src proto.Message) {
	xxx_messageInfo_HealthCheckRequest.Merge(m, src)
}
func (m *HealthCheckRequest) XXX_Size() int {
	return m.Size()
}
func (m *HealthCheckRequest) XXX_DiscardUnknown() {
	xxx_messageInfo_HealthCheckRequest.DiscardUnknown(m)
}

var xxx_messageInfo_HealthCheckRequest proto.InternalMessageInfo

type HealthCheckReply struct {
	Title   string `protobuf:"bytes,1,opt,name=title,proto3" json:"title,omitempty"`
	Version string `protobuf:"bytes,2,opt,name=version,proto3" json:"version,omitempty"`
	Commit  string `protobuf:"bytes,3,opt,name=commit,proto3" json:"commit,omitempty"`
}

func (m *HealthCheckReply) Reset()      { *m = HealthCheckReply{} }
func (*HealthCheckReply) ProtoMessage() {}
func (*HealthCheckReply) Descriptor() ([]byte, []int) {
	return fileDescriptor_e0e9c33bd10e4006, []int{1}
}
func (m *HealthCheckReply) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *HealthCheckReply) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_HealthCheckReply.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *HealthCheckReply) XXX_Merge(src proto.Message) {
	xxx_messageInfo_HealthCheckReply.Merge(m, src)
}
func (m *HealthCheckReply) XXX_Size() int {
	return m.Size()
}
func (m *HealthCheckReply) XXX_DiscardUnknown() {
	xxx_messageInfo_HealthCheckReply.DiscardUnknown(m)
}

var xxx_messageInfo_HealthCheckReply proto.InternalMessageInfo

func (m *HealthCheckReply) GetTitle() string {
	if m != nil {
		return m.Title
	}
	return ""
}

func (m *HealthCheckReply) GetVersion() string {
	if m != nil {
		return m.Version
	}
	return ""
}

func (m *HealthCheckReply) GetCommit() string {
	if m != nil {
		return m.Commit
	}
	return ""
}

func init() {
	proto.RegisterType((*HealthCheckRequest)(nil), "qdrant.HealthCheckRequest")
	proto.RegisterType((*HealthCheckReply)(nil), "qdrant.HealthCheckReply")
}

func init() { proto.RegisterFile("qdrant/qdrant.proto", fileDescriptor_e0e9c33bd10e4006) }

var fileDescriptor_e0e9c33bd10e4006 = []byte{
	// 288 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x6c, 0x91, 0xc1, 0x4a, 0xeb, 0x40,
	0x14, 0x86, 0x67, 0xee, 0xc5, 0x88, 0xe3, 0x46, 0xc6, 0x22, 0x21, 0xc2, 0xa1, 0x64, 0xe5, 0xc6,
	0x14, 0xf4, 0x0d, 0x14, 0xc1, 0x9d, 0xd8, 0x95, 0x74, 0x23, 0xe9, 0x38, 0x34, 0x83, 0x93, 0x4c,
	0x9a, 0x39, 0x2d, 0x74, 0xe7, 0x23, 0xf8, 0x18, 0x3e, 0x8a, 0xcb, 0x2c, 0xbb, 0x34, 0x93, 0x8d,
	0xcb, 0x3e, 0x82, 0x90, 0x8c, 0xa2, 0xd5, 0xd5, 0xf0, 0x9d, 0xef, 0xc0, 0xfc, 0xfc, 0x87, 0x1d,
	0xce, 0x1f, 0xaa, 0xb4, 0xc0, 0x51, 0xff, 0x24, 0x65, 0x65, 0xd0, 0xf0, 0xa0, 0xa7, 0x68, 0xe8,
	0xa5, 0x30, 0x5a, 0x4b, 0x81, 0xca, 0x14, 0xf6, 0xde, 0xca, 0x6a, 0xa9, 0x84, 0xec, 0x37, 0xa3,
	0x63, 0xbf, 0x51, 0x1a, 0x55, 0xe0, 0xb6, 0x04, 0x2f, 0x6d, 0x91, 0x96, 0x36, 0x33, 0xdb, 0x3e,
	0x1e, 0x30, 0x7e, 0x2d, 0x53, 0x8d, 0xd9, 0x65, 0x26, 0xc5, 0xe3, 0x58, 0xce, 0x17, 0xd2, 0x62,
	0x3c, 0x61, 0x07, 0x3f, 0xa6, 0xa5, 0x5e, 0xf1, 0x01, 0xdb, 0x41, 0x85, 0x5a, 0x86, 0x74, 0x48,
	0x4f, 0xf6, 0xc6, 0x3d, 0xf0, 0x90, 0xed, 0x2e, 0x65, 0x65, 0x95, 0x29, 0xc2, 0x7f, 0xdd, 0xfc,
	0x13, 0xf9, 0x11, 0x0b, 0x84, 0xc9, 0x73, 0x85, 0xe1, 0xff, 0x4e, 0x78, 0x3a, 0xbb, 0x61, 0xc1,
	0x6d, 0x97, 0x89, 0x5f, 0xb1, 0xfd, 0x6f, 0xbf, 0xf0, 0x28, 0xf1, 0x05, 0xfc, 0x0e, 0x14, 0x85,
	0x7f, 0xba, 0x52, 0xaf, 0x62, 0x72, 0x71, 0x57, 0x37, 0x40, 0xd6, 0x0d, 0x90, 0x4d, 0x03, 0xf4,
	0xc9, 0x01, 0x7d, 0x71, 0x40, 0x5f, 0x1d, 0xd0, 0xda, 0x01, 0x7d, 0x73, 0x40, 0xdf, 0x1d, 0x90,
	0x8d, 0x03, 0xfa, 0xdc, 0x02, 0xa9, 0x5b, 0x20, 0xeb, 0x16, 0xc8, 0x24, 0x9e, 0x29, 0xcc, 0x16,
	0xd3, 0x44, 0x98, 0xdc, 0x17, 0x3f, 0x9a, 0x99, 0x53, 0xa1, 0x95, 0xfc, 0xba, 0xc4, 0x34, 0xe8,
	0x3a, 0x3a, 0xff, 0x08, 0x00, 0x00, 0xff, 0xff, 0x20, 0x32, 0x1b, 0x3f, 0xa1, 0x01, 0x00, 0x00,
}

func (this *HealthCheckRequest) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*HealthCheckRequest)
	if !ok {
		that2, ok := that.(HealthCheckRequest)
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
func (this *HealthCheckReply) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*HealthCheckReply)
	if !ok {
		that2, ok := that.(HealthCheckReply)
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
	if this.Title != that1.Title {
		return false
	}
	if this.Version != that1.Version {
		return false
	}
	if this.Commit != that1.Commit {
		return false
	}
	return true
}
func (this *HealthCheckRequest) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 4)
	s = append(s, "&qdrant.HealthCheckRequest{")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *HealthCheckReply) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 7)
	s = append(s, "&qdrant.HealthCheckReply{")
	s = append(s, "Title: "+fmt.Sprintf("%#v", this.Title)+",\n")
	s = append(s, "Version: "+fmt.Sprintf("%#v", this.Version)+",\n")
	s = append(s, "Commit: "+fmt.Sprintf("%#v", this.Commit)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringQdrant(v interface{}, typ string) string {
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

// QdrantClient is the client API for Qdrant service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type QdrantClient interface {
	HealthCheck(ctx context.Context, in *HealthCheckRequest, opts ...grpc.CallOption) (*HealthCheckReply, error)
}

type qdrantClient struct {
	cc *grpc.ClientConn
}

func NewQdrantClient(cc *grpc.ClientConn) QdrantClient {
	return &qdrantClient{cc}
}

func (c *qdrantClient) HealthCheck(ctx context.Context, in *HealthCheckRequest, opts ...grpc.CallOption) (*HealthCheckReply, error) {
	out := new(HealthCheckReply)
	err := c.cc.Invoke(ctx, "/qdrant.Qdrant/HealthCheck", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// QdrantServer is the server API for Qdrant service.
type QdrantServer interface {
	HealthCheck(context.Context, *HealthCheckRequest) (*HealthCheckReply, error)
}

// UnimplementedQdrantServer can be embedded to have forward compatible implementations.
type UnimplementedQdrantServer struct {
}

func (*UnimplementedQdrantServer) HealthCheck(ctx context.Context, req *HealthCheckRequest) (*HealthCheckReply, error) {
	return nil, status.Errorf(codes.Unimplemented, "method HealthCheck not implemented")
}

func RegisterQdrantServer(s *grpc.Server, srv QdrantServer) {
	s.RegisterService(&_Qdrant_serviceDesc, srv)
}

func _Qdrant_HealthCheck_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(HealthCheckRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(QdrantServer).HealthCheck(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/qdrant.Qdrant/HealthCheck",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(QdrantServer).HealthCheck(ctx, req.(*HealthCheckRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _Qdrant_serviceDesc = grpc.ServiceDesc{
	ServiceName: "qdrant.Qdrant",
	HandlerType: (*QdrantServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "HealthCheck",
			Handler:    _Qdrant_HealthCheck_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "qdrant/qdrant.proto",
}

func (m *HealthCheckRequest) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *HealthCheckRequest) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *HealthCheckRequest) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	return len(dAtA) - i, nil
}

func (m *HealthCheckReply) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *HealthCheckReply) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *HealthCheckReply) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Commit) > 0 {
		i -= len(m.Commit)
		copy(dAtA[i:], m.Commit)
		i = encodeVarintQdrant(dAtA, i, uint64(len(m.Commit)))
		i--
		dAtA[i] = 0x1a
	}
	if len(m.Version) > 0 {
		i -= len(m.Version)
		copy(dAtA[i:], m.Version)
		i = encodeVarintQdrant(dAtA, i, uint64(len(m.Version)))
		i--
		dAtA[i] = 0x12
	}
	if len(m.Title) > 0 {
		i -= len(m.Title)
		copy(dAtA[i:], m.Title)
		i = encodeVarintQdrant(dAtA, i, uint64(len(m.Title)))
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintQdrant(dAtA []byte, offset int, v uint64) int {
	offset -= sovQdrant(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *HealthCheckRequest) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	return n
}

func (m *HealthCheckReply) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	l = len(m.Title)
	if l > 0 {
		n += 1 + l + sovQdrant(uint64(l))
	}
	l = len(m.Version)
	if l > 0 {
		n += 1 + l + sovQdrant(uint64(l))
	}
	l = len(m.Commit)
	if l > 0 {
		n += 1 + l + sovQdrant(uint64(l))
	}
	return n
}

func sovQdrant(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozQdrant(x uint64) (n int) {
	return sovQdrant(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *HealthCheckRequest) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&HealthCheckRequest{`,
		`}`,
	}, "")
	return s
}
func (this *HealthCheckReply) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&HealthCheckReply{`,
		`Title:` + fmt.Sprintf("%v", this.Title) + `,`,
		`Version:` + fmt.Sprintf("%v", this.Version) + `,`,
		`Commit:` + fmt.Sprintf("%v", this.Commit) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringQdrant(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *HealthCheckRequest) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowQdrant
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
			return fmt.Errorf("proto: HealthCheckRequest: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: HealthCheckRequest: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		default:
			iNdEx = preIndex
			skippy, err := skipQdrant(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthQdrant
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
func (m *HealthCheckReply) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowQdrant
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
			return fmt.Errorf("proto: HealthCheckReply: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: HealthCheckReply: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Title", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowQdrant
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
				return ErrInvalidLengthQdrant
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthQdrant
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Title = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Version", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowQdrant
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
				return ErrInvalidLengthQdrant
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthQdrant
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Version = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 3:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Commit", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowQdrant
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
				return ErrInvalidLengthQdrant
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthQdrant
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Commit = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipQdrant(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthQdrant
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
func skipQdrant(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowQdrant
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
					return 0, ErrIntOverflowQdrant
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
					return 0, ErrIntOverflowQdrant
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
				return 0, ErrInvalidLengthQdrant
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupQdrant
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthQdrant
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthQdrant        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowQdrant          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupQdrant = fmt.Errorf("proto: unexpected end of group")
)
