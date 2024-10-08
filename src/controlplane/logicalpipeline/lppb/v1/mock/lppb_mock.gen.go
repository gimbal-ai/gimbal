// Code generated by MockGen. DO NOT EDIT.
// Source: lppb.pb.go
//
// Generated by this command:
//
//	mockgen -source=lppb.pb.go -destination=mock/lppb_mock.gen.go LogicalPipelineService
//
// Package mock_lppb is a generated GoMock package.
package mock_lppb

import (
	context "context"
	reflect "reflect"

	lppb "gimletlabs.ai/gimlet/src/controlplane/logicalpipeline/lppb/v1"
	gomock "go.uber.org/mock/gomock"
	grpc "google.golang.org/grpc"
)

// MockLogicalPipelineServiceClient is a mock of LogicalPipelineServiceClient interface.
type MockLogicalPipelineServiceClient struct {
	ctrl     *gomock.Controller
	recorder *MockLogicalPipelineServiceClientMockRecorder
}

// MockLogicalPipelineServiceClientMockRecorder is the mock recorder for MockLogicalPipelineServiceClient.
type MockLogicalPipelineServiceClientMockRecorder struct {
	mock *MockLogicalPipelineServiceClient
}

// NewMockLogicalPipelineServiceClient creates a new mock instance.
func NewMockLogicalPipelineServiceClient(ctrl *gomock.Controller) *MockLogicalPipelineServiceClient {
	mock := &MockLogicalPipelineServiceClient{ctrl: ctrl}
	mock.recorder = &MockLogicalPipelineServiceClientMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockLogicalPipelineServiceClient) EXPECT() *MockLogicalPipelineServiceClientMockRecorder {
	return m.recorder
}

// CreateLogicalPipeline mocks base method.
func (m *MockLogicalPipelineServiceClient) CreateLogicalPipeline(ctx context.Context, in *lppb.CreateLogicalPipelineRequest, opts ...grpc.CallOption) (*lppb.CreateLogicalPipelineResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "CreateLogicalPipeline", varargs...)
	ret0, _ := ret[0].(*lppb.CreateLogicalPipelineResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// CreateLogicalPipeline indicates an expected call of CreateLogicalPipeline.
func (mr *MockLogicalPipelineServiceClientMockRecorder) CreateLogicalPipeline(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateLogicalPipeline", reflect.TypeOf((*MockLogicalPipelineServiceClient)(nil).CreateLogicalPipeline), varargs...)
}

// GetLogicalPipeline mocks base method.
func (m *MockLogicalPipelineServiceClient) GetLogicalPipeline(ctx context.Context, in *lppb.GetLogicalPipelineRequest, opts ...grpc.CallOption) (*lppb.GetLogicalPipelineResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "GetLogicalPipeline", varargs...)
	ret0, _ := ret[0].(*lppb.GetLogicalPipelineResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetLogicalPipeline indicates an expected call of GetLogicalPipeline.
func (mr *MockLogicalPipelineServiceClientMockRecorder) GetLogicalPipeline(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetLogicalPipeline", reflect.TypeOf((*MockLogicalPipelineServiceClient)(nil).GetLogicalPipeline), varargs...)
}

// ListLogicalPipelines mocks base method.
func (m *MockLogicalPipelineServiceClient) ListLogicalPipelines(ctx context.Context, in *lppb.ListLogicalPipelinesRequest, opts ...grpc.CallOption) (*lppb.ListLogicalPipelinesResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "ListLogicalPipelines", varargs...)
	ret0, _ := ret[0].(*lppb.ListLogicalPipelinesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListLogicalPipelines indicates an expected call of ListLogicalPipelines.
func (mr *MockLogicalPipelineServiceClientMockRecorder) ListLogicalPipelines(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListLogicalPipelines", reflect.TypeOf((*MockLogicalPipelineServiceClient)(nil).ListLogicalPipelines), varargs...)
}

// ParseLogicalPipelineYAML mocks base method.
func (m *MockLogicalPipelineServiceClient) ParseLogicalPipelineYAML(ctx context.Context, in *lppb.ParseLogicalPipelineYAMLRequest, opts ...grpc.CallOption) (*lppb.ParseLogicalPipelineYAMLResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "ParseLogicalPipelineYAML", varargs...)
	ret0, _ := ret[0].(*lppb.ParseLogicalPipelineYAMLResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ParseLogicalPipelineYAML indicates an expected call of ParseLogicalPipelineYAML.
func (mr *MockLogicalPipelineServiceClientMockRecorder) ParseLogicalPipelineYAML(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ParseLogicalPipelineYAML", reflect.TypeOf((*MockLogicalPipelineServiceClient)(nil).ParseLogicalPipelineYAML), varargs...)
}

// MockLogicalPipelineServiceServer is a mock of LogicalPipelineServiceServer interface.
type MockLogicalPipelineServiceServer struct {
	ctrl     *gomock.Controller
	recorder *MockLogicalPipelineServiceServerMockRecorder
}

// MockLogicalPipelineServiceServerMockRecorder is the mock recorder for MockLogicalPipelineServiceServer.
type MockLogicalPipelineServiceServerMockRecorder struct {
	mock *MockLogicalPipelineServiceServer
}

// NewMockLogicalPipelineServiceServer creates a new mock instance.
func NewMockLogicalPipelineServiceServer(ctrl *gomock.Controller) *MockLogicalPipelineServiceServer {
	mock := &MockLogicalPipelineServiceServer{ctrl: ctrl}
	mock.recorder = &MockLogicalPipelineServiceServerMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockLogicalPipelineServiceServer) EXPECT() *MockLogicalPipelineServiceServerMockRecorder {
	return m.recorder
}

// CreateLogicalPipeline mocks base method.
func (m *MockLogicalPipelineServiceServer) CreateLogicalPipeline(arg0 context.Context, arg1 *lppb.CreateLogicalPipelineRequest) (*lppb.CreateLogicalPipelineResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "CreateLogicalPipeline", arg0, arg1)
	ret0, _ := ret[0].(*lppb.CreateLogicalPipelineResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// CreateLogicalPipeline indicates an expected call of CreateLogicalPipeline.
func (mr *MockLogicalPipelineServiceServerMockRecorder) CreateLogicalPipeline(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateLogicalPipeline", reflect.TypeOf((*MockLogicalPipelineServiceServer)(nil).CreateLogicalPipeline), arg0, arg1)
}

// GetLogicalPipeline mocks base method.
func (m *MockLogicalPipelineServiceServer) GetLogicalPipeline(arg0 context.Context, arg1 *lppb.GetLogicalPipelineRequest) (*lppb.GetLogicalPipelineResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetLogicalPipeline", arg0, arg1)
	ret0, _ := ret[0].(*lppb.GetLogicalPipelineResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetLogicalPipeline indicates an expected call of GetLogicalPipeline.
func (mr *MockLogicalPipelineServiceServerMockRecorder) GetLogicalPipeline(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetLogicalPipeline", reflect.TypeOf((*MockLogicalPipelineServiceServer)(nil).GetLogicalPipeline), arg0, arg1)
}

// ListLogicalPipelines mocks base method.
func (m *MockLogicalPipelineServiceServer) ListLogicalPipelines(arg0 context.Context, arg1 *lppb.ListLogicalPipelinesRequest) (*lppb.ListLogicalPipelinesResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ListLogicalPipelines", arg0, arg1)
	ret0, _ := ret[0].(*lppb.ListLogicalPipelinesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListLogicalPipelines indicates an expected call of ListLogicalPipelines.
func (mr *MockLogicalPipelineServiceServerMockRecorder) ListLogicalPipelines(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListLogicalPipelines", reflect.TypeOf((*MockLogicalPipelineServiceServer)(nil).ListLogicalPipelines), arg0, arg1)
}

// ParseLogicalPipelineYAML mocks base method.
func (m *MockLogicalPipelineServiceServer) ParseLogicalPipelineYAML(arg0 context.Context, arg1 *lppb.ParseLogicalPipelineYAMLRequest) (*lppb.ParseLogicalPipelineYAMLResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ParseLogicalPipelineYAML", arg0, arg1)
	ret0, _ := ret[0].(*lppb.ParseLogicalPipelineYAMLResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ParseLogicalPipelineYAML indicates an expected call of ParseLogicalPipelineYAML.
func (mr *MockLogicalPipelineServiceServerMockRecorder) ParseLogicalPipelineYAML(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ParseLogicalPipelineYAML", reflect.TypeOf((*MockLogicalPipelineServiceServer)(nil).ParseLogicalPipelineYAML), arg0, arg1)
}
