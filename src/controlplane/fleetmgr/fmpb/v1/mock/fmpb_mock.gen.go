// Code generated by MockGen. DO NOT EDIT.
// Source: fmpb.pb.go
//
// Generated by this command:
//
//	mockgen -source=fmpb.pb.go -destination=mock/fmpb_mock.gen.go FleetMgrService,FleetMgrEdgeService
//
// Package mock_fmpb is a generated GoMock package.
package mock_fmpb

import (
	context "context"
	reflect "reflect"

	fmpb "gimletlabs.ai/gimlet/src/controlplane/fleetmgr/fmpb/v1"
	gomock "go.uber.org/mock/gomock"
	grpc "google.golang.org/grpc"
)

// MockFleetMgrServiceClient is a mock of FleetMgrServiceClient interface.
type MockFleetMgrServiceClient struct {
	ctrl     *gomock.Controller
	recorder *MockFleetMgrServiceClientMockRecorder
}

// MockFleetMgrServiceClientMockRecorder is the mock recorder for MockFleetMgrServiceClient.
type MockFleetMgrServiceClientMockRecorder struct {
	mock *MockFleetMgrServiceClient
}

// NewMockFleetMgrServiceClient creates a new mock instance.
func NewMockFleetMgrServiceClient(ctrl *gomock.Controller) *MockFleetMgrServiceClient {
	mock := &MockFleetMgrServiceClient{ctrl: ctrl}
	mock.recorder = &MockFleetMgrServiceClientMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockFleetMgrServiceClient) EXPECT() *MockFleetMgrServiceClientMockRecorder {
	return m.recorder
}

// CreateFleet mocks base method.
func (m *MockFleetMgrServiceClient) CreateFleet(ctx context.Context, in *fmpb.CreateFleetRequest, opts ...grpc.CallOption) (*fmpb.CreateFleetResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "CreateFleet", varargs...)
	ret0, _ := ret[0].(*fmpb.CreateFleetResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// CreateFleet indicates an expected call of CreateFleet.
func (mr *MockFleetMgrServiceClientMockRecorder) CreateFleet(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateFleet", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).CreateFleet), varargs...)
}

// DeleteDefaultTag mocks base method.
func (m *MockFleetMgrServiceClient) DeleteDefaultTag(ctx context.Context, in *fmpb.DeleteDefaultTagRequest, opts ...grpc.CallOption) (*fmpb.DeleteDefaultTagResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "DeleteDefaultTag", varargs...)
	ret0, _ := ret[0].(*fmpb.DeleteDefaultTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeleteDefaultTag indicates an expected call of DeleteDefaultTag.
func (mr *MockFleetMgrServiceClientMockRecorder) DeleteDefaultTag(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteDefaultTag", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).DeleteDefaultTag), varargs...)
}

// GetDefaultTags mocks base method.
func (m *MockFleetMgrServiceClient) GetDefaultTags(ctx context.Context, in *fmpb.GetDefaultTagsRequest, opts ...grpc.CallOption) (*fmpb.GetDefaultTagsResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "GetDefaultTags", varargs...)
	ret0, _ := ret[0].(*fmpb.GetDefaultTagsResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetDefaultTags indicates an expected call of GetDefaultTags.
func (mr *MockFleetMgrServiceClientMockRecorder) GetDefaultTags(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetDefaultTags", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).GetDefaultTags), varargs...)
}

// GetFleet mocks base method.
func (m *MockFleetMgrServiceClient) GetFleet(ctx context.Context, in *fmpb.GetFleetRequest, opts ...grpc.CallOption) (*fmpb.GetFleetResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "GetFleet", varargs...)
	ret0, _ := ret[0].(*fmpb.GetFleetResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetFleet indicates an expected call of GetFleet.
func (mr *MockFleetMgrServiceClientMockRecorder) GetFleet(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetFleet", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).GetFleet), varargs...)
}

// GetFleetByName mocks base method.
func (m *MockFleetMgrServiceClient) GetFleetByName(ctx context.Context, in *fmpb.GetFleetByNameRequest, opts ...grpc.CallOption) (*fmpb.GetFleetByNameResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "GetFleetByName", varargs...)
	ret0, _ := ret[0].(*fmpb.GetFleetByNameResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetFleetByName indicates an expected call of GetFleetByName.
func (mr *MockFleetMgrServiceClientMockRecorder) GetFleetByName(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetFleetByName", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).GetFleetByName), varargs...)
}

// ListFleets mocks base method.
func (m *MockFleetMgrServiceClient) ListFleets(ctx context.Context, in *fmpb.ListFleetsRequest, opts ...grpc.CallOption) (*fmpb.ListFleetsResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "ListFleets", varargs...)
	ret0, _ := ret[0].(*fmpb.ListFleetsResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListFleets indicates an expected call of ListFleets.
func (mr *MockFleetMgrServiceClientMockRecorder) ListFleets(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListFleets", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).ListFleets), varargs...)
}

// UpdateFleet mocks base method.
func (m *MockFleetMgrServiceClient) UpdateFleet(ctx context.Context, in *fmpb.UpdateFleetRequest, opts ...grpc.CallOption) (*fmpb.UpdateFleetResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "UpdateFleet", varargs...)
	ret0, _ := ret[0].(*fmpb.UpdateFleetResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpdateFleet indicates an expected call of UpdateFleet.
func (mr *MockFleetMgrServiceClientMockRecorder) UpdateFleet(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpdateFleet", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).UpdateFleet), varargs...)
}

// UpsertDefaultTag mocks base method.
func (m *MockFleetMgrServiceClient) UpsertDefaultTag(ctx context.Context, in *fmpb.UpsertDefaultTagRequest, opts ...grpc.CallOption) (*fmpb.UpsertDefaultTagResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "UpsertDefaultTag", varargs...)
	ret0, _ := ret[0].(*fmpb.UpsertDefaultTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpsertDefaultTag indicates an expected call of UpsertDefaultTag.
func (mr *MockFleetMgrServiceClientMockRecorder) UpsertDefaultTag(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpsertDefaultTag", reflect.TypeOf((*MockFleetMgrServiceClient)(nil).UpsertDefaultTag), varargs...)
}

// MockFleetMgrServiceServer is a mock of FleetMgrServiceServer interface.
type MockFleetMgrServiceServer struct {
	ctrl     *gomock.Controller
	recorder *MockFleetMgrServiceServerMockRecorder
}

// MockFleetMgrServiceServerMockRecorder is the mock recorder for MockFleetMgrServiceServer.
type MockFleetMgrServiceServerMockRecorder struct {
	mock *MockFleetMgrServiceServer
}

// NewMockFleetMgrServiceServer creates a new mock instance.
func NewMockFleetMgrServiceServer(ctrl *gomock.Controller) *MockFleetMgrServiceServer {
	mock := &MockFleetMgrServiceServer{ctrl: ctrl}
	mock.recorder = &MockFleetMgrServiceServerMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockFleetMgrServiceServer) EXPECT() *MockFleetMgrServiceServerMockRecorder {
	return m.recorder
}

// CreateFleet mocks base method.
func (m *MockFleetMgrServiceServer) CreateFleet(arg0 context.Context, arg1 *fmpb.CreateFleetRequest) (*fmpb.CreateFleetResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "CreateFleet", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.CreateFleetResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// CreateFleet indicates an expected call of CreateFleet.
func (mr *MockFleetMgrServiceServerMockRecorder) CreateFleet(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "CreateFleet", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).CreateFleet), arg0, arg1)
}

// DeleteDefaultTag mocks base method.
func (m *MockFleetMgrServiceServer) DeleteDefaultTag(arg0 context.Context, arg1 *fmpb.DeleteDefaultTagRequest) (*fmpb.DeleteDefaultTagResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeleteDefaultTag", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.DeleteDefaultTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeleteDefaultTag indicates an expected call of DeleteDefaultTag.
func (mr *MockFleetMgrServiceServerMockRecorder) DeleteDefaultTag(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteDefaultTag", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).DeleteDefaultTag), arg0, arg1)
}

// GetDefaultTags mocks base method.
func (m *MockFleetMgrServiceServer) GetDefaultTags(arg0 context.Context, arg1 *fmpb.GetDefaultTagsRequest) (*fmpb.GetDefaultTagsResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetDefaultTags", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.GetDefaultTagsResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetDefaultTags indicates an expected call of GetDefaultTags.
func (mr *MockFleetMgrServiceServerMockRecorder) GetDefaultTags(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetDefaultTags", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).GetDefaultTags), arg0, arg1)
}

// GetFleet mocks base method.
func (m *MockFleetMgrServiceServer) GetFleet(arg0 context.Context, arg1 *fmpb.GetFleetRequest) (*fmpb.GetFleetResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetFleet", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.GetFleetResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetFleet indicates an expected call of GetFleet.
func (mr *MockFleetMgrServiceServerMockRecorder) GetFleet(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetFleet", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).GetFleet), arg0, arg1)
}

// GetFleetByName mocks base method.
func (m *MockFleetMgrServiceServer) GetFleetByName(arg0 context.Context, arg1 *fmpb.GetFleetByNameRequest) (*fmpb.GetFleetByNameResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetFleetByName", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.GetFleetByNameResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetFleetByName indicates an expected call of GetFleetByName.
func (mr *MockFleetMgrServiceServerMockRecorder) GetFleetByName(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetFleetByName", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).GetFleetByName), arg0, arg1)
}

// ListFleets mocks base method.
func (m *MockFleetMgrServiceServer) ListFleets(arg0 context.Context, arg1 *fmpb.ListFleetsRequest) (*fmpb.ListFleetsResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ListFleets", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.ListFleetsResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListFleets indicates an expected call of ListFleets.
func (mr *MockFleetMgrServiceServerMockRecorder) ListFleets(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListFleets", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).ListFleets), arg0, arg1)
}

// UpdateFleet mocks base method.
func (m *MockFleetMgrServiceServer) UpdateFleet(arg0 context.Context, arg1 *fmpb.UpdateFleetRequest) (*fmpb.UpdateFleetResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "UpdateFleet", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.UpdateFleetResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpdateFleet indicates an expected call of UpdateFleet.
func (mr *MockFleetMgrServiceServerMockRecorder) UpdateFleet(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpdateFleet", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).UpdateFleet), arg0, arg1)
}

// UpsertDefaultTag mocks base method.
func (m *MockFleetMgrServiceServer) UpsertDefaultTag(arg0 context.Context, arg1 *fmpb.UpsertDefaultTagRequest) (*fmpb.UpsertDefaultTagResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "UpsertDefaultTag", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.UpsertDefaultTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpsertDefaultTag indicates an expected call of UpsertDefaultTag.
func (mr *MockFleetMgrServiceServerMockRecorder) UpsertDefaultTag(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpsertDefaultTag", reflect.TypeOf((*MockFleetMgrServiceServer)(nil).UpsertDefaultTag), arg0, arg1)
}

// MockFleetMgrEdgeServiceClient is a mock of FleetMgrEdgeServiceClient interface.
type MockFleetMgrEdgeServiceClient struct {
	ctrl     *gomock.Controller
	recorder *MockFleetMgrEdgeServiceClientMockRecorder
}

// MockFleetMgrEdgeServiceClientMockRecorder is the mock recorder for MockFleetMgrEdgeServiceClient.
type MockFleetMgrEdgeServiceClientMockRecorder struct {
	mock *MockFleetMgrEdgeServiceClient
}

// NewMockFleetMgrEdgeServiceClient creates a new mock instance.
func NewMockFleetMgrEdgeServiceClient(ctrl *gomock.Controller) *MockFleetMgrEdgeServiceClient {
	mock := &MockFleetMgrEdgeServiceClient{ctrl: ctrl}
	mock.recorder = &MockFleetMgrEdgeServiceClientMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockFleetMgrEdgeServiceClient) EXPECT() *MockFleetMgrEdgeServiceClientMockRecorder {
	return m.recorder
}

// AssociateTagsWithDeployKey mocks base method.
func (m *MockFleetMgrEdgeServiceClient) AssociateTagsWithDeployKey(ctx context.Context, in *fmpb.AssociateTagsWithDeployKeyRequest, opts ...grpc.CallOption) (*fmpb.AssociateTagsWithDeployKeyResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "AssociateTagsWithDeployKey", varargs...)
	ret0, _ := ret[0].(*fmpb.AssociateTagsWithDeployKeyResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// AssociateTagsWithDeployKey indicates an expected call of AssociateTagsWithDeployKey.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) AssociateTagsWithDeployKey(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AssociateTagsWithDeployKey", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).AssociateTagsWithDeployKey), varargs...)
}

// DeleteDevices mocks base method.
func (m *MockFleetMgrEdgeServiceClient) DeleteDevices(ctx context.Context, in *fmpb.DeleteDevicesRequest, opts ...grpc.CallOption) (*fmpb.DeleteDevicesResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "DeleteDevices", varargs...)
	ret0, _ := ret[0].(*fmpb.DeleteDevicesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeleteDevices indicates an expected call of DeleteDevices.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) DeleteDevices(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteDevices", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).DeleteDevices), varargs...)
}

// DeleteTag mocks base method.
func (m *MockFleetMgrEdgeServiceClient) DeleteTag(ctx context.Context, in *fmpb.DeleteTagRequest, opts ...grpc.CallOption) (*fmpb.DeleteTagResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "DeleteTag", varargs...)
	ret0, _ := ret[0].(*fmpb.DeleteTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeleteTag indicates an expected call of DeleteTag.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) DeleteTag(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteTag", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).DeleteTag), varargs...)
}

// GetDevice mocks base method.
func (m *MockFleetMgrEdgeServiceClient) GetDevice(ctx context.Context, in *fmpb.GetDeviceRequest, opts ...grpc.CallOption) (*fmpb.GetDeviceResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "GetDevice", varargs...)
	ret0, _ := ret[0].(*fmpb.GetDeviceResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetDevice indicates an expected call of GetDevice.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) GetDevice(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetDevice", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).GetDevice), varargs...)
}

// GetTags mocks base method.
func (m *MockFleetMgrEdgeServiceClient) GetTags(ctx context.Context, in *fmpb.GetTagsRequest, opts ...grpc.CallOption) (*fmpb.GetTagsResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "GetTags", varargs...)
	ret0, _ := ret[0].(*fmpb.GetTagsResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetTags indicates an expected call of GetTags.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) GetTags(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetTags", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).GetTags), varargs...)
}

// ListDevices mocks base method.
func (m *MockFleetMgrEdgeServiceClient) ListDevices(ctx context.Context, in *fmpb.ListDevicesRequest, opts ...grpc.CallOption) (*fmpb.ListDevicesResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "ListDevices", varargs...)
	ret0, _ := ret[0].(*fmpb.ListDevicesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListDevices indicates an expected call of ListDevices.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) ListDevices(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListDevices", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).ListDevices), varargs...)
}

// ListTagsAssociatedWithDeployKey mocks base method.
func (m *MockFleetMgrEdgeServiceClient) ListTagsAssociatedWithDeployKey(ctx context.Context, in *fmpb.ListTagsAssociatedWithDeployKeyRequest, opts ...grpc.CallOption) (*fmpb.ListTagsAssociatedWithDeployKeyResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "ListTagsAssociatedWithDeployKey", varargs...)
	ret0, _ := ret[0].(*fmpb.ListTagsAssociatedWithDeployKeyResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListTagsAssociatedWithDeployKey indicates an expected call of ListTagsAssociatedWithDeployKey.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) ListTagsAssociatedWithDeployKey(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListTagsAssociatedWithDeployKey", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).ListTagsAssociatedWithDeployKey), varargs...)
}

// Register mocks base method.
func (m *MockFleetMgrEdgeServiceClient) Register(ctx context.Context, in *fmpb.RegisterRequest, opts ...grpc.CallOption) (*fmpb.RegisterResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "Register", varargs...)
	ret0, _ := ret[0].(*fmpb.RegisterResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// Register indicates an expected call of Register.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) Register(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Register", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).Register), varargs...)
}

// SetDeviceCapabilities mocks base method.
func (m *MockFleetMgrEdgeServiceClient) SetDeviceCapabilities(ctx context.Context, in *fmpb.SetDeviceCapabilitiesRequest, opts ...grpc.CallOption) (*fmpb.SetDeviceCapabilitiesResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "SetDeviceCapabilities", varargs...)
	ret0, _ := ret[0].(*fmpb.SetDeviceCapabilitiesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// SetDeviceCapabilities indicates an expected call of SetDeviceCapabilities.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) SetDeviceCapabilities(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetDeviceCapabilities", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).SetDeviceCapabilities), varargs...)
}

// UpdateDevice mocks base method.
func (m *MockFleetMgrEdgeServiceClient) UpdateDevice(ctx context.Context, in *fmpb.UpdateDeviceRequest, opts ...grpc.CallOption) (*fmpb.UpdateDeviceResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "UpdateDevice", varargs...)
	ret0, _ := ret[0].(*fmpb.UpdateDeviceResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpdateDevice indicates an expected call of UpdateDevice.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) UpdateDevice(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpdateDevice", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).UpdateDevice), varargs...)
}

// UpdateStatus mocks base method.
func (m *MockFleetMgrEdgeServiceClient) UpdateStatus(ctx context.Context, in *fmpb.UpdateStatusRequest, opts ...grpc.CallOption) (*fmpb.UpdateStatusResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "UpdateStatus", varargs...)
	ret0, _ := ret[0].(*fmpb.UpdateStatusResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpdateStatus indicates an expected call of UpdateStatus.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) UpdateStatus(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpdateStatus", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).UpdateStatus), varargs...)
}

// UpsertTag mocks base method.
func (m *MockFleetMgrEdgeServiceClient) UpsertTag(ctx context.Context, in *fmpb.UpsertTagRequest, opts ...grpc.CallOption) (*fmpb.UpsertTagResponse, error) {
	m.ctrl.T.Helper()
	varargs := []any{ctx, in}
	for _, a := range opts {
		varargs = append(varargs, a)
	}
	ret := m.ctrl.Call(m, "UpsertTag", varargs...)
	ret0, _ := ret[0].(*fmpb.UpsertTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpsertTag indicates an expected call of UpsertTag.
func (mr *MockFleetMgrEdgeServiceClientMockRecorder) UpsertTag(ctx, in any, opts ...any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	varargs := append([]any{ctx, in}, opts...)
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpsertTag", reflect.TypeOf((*MockFleetMgrEdgeServiceClient)(nil).UpsertTag), varargs...)
}

// MockFleetMgrEdgeServiceServer is a mock of FleetMgrEdgeServiceServer interface.
type MockFleetMgrEdgeServiceServer struct {
	ctrl     *gomock.Controller
	recorder *MockFleetMgrEdgeServiceServerMockRecorder
}

// MockFleetMgrEdgeServiceServerMockRecorder is the mock recorder for MockFleetMgrEdgeServiceServer.
type MockFleetMgrEdgeServiceServerMockRecorder struct {
	mock *MockFleetMgrEdgeServiceServer
}

// NewMockFleetMgrEdgeServiceServer creates a new mock instance.
func NewMockFleetMgrEdgeServiceServer(ctrl *gomock.Controller) *MockFleetMgrEdgeServiceServer {
	mock := &MockFleetMgrEdgeServiceServer{ctrl: ctrl}
	mock.recorder = &MockFleetMgrEdgeServiceServerMockRecorder{mock}
	return mock
}

// EXPECT returns an object that allows the caller to indicate expected use.
func (m *MockFleetMgrEdgeServiceServer) EXPECT() *MockFleetMgrEdgeServiceServerMockRecorder {
	return m.recorder
}

// AssociateTagsWithDeployKey mocks base method.
func (m *MockFleetMgrEdgeServiceServer) AssociateTagsWithDeployKey(arg0 context.Context, arg1 *fmpb.AssociateTagsWithDeployKeyRequest) (*fmpb.AssociateTagsWithDeployKeyResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "AssociateTagsWithDeployKey", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.AssociateTagsWithDeployKeyResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// AssociateTagsWithDeployKey indicates an expected call of AssociateTagsWithDeployKey.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) AssociateTagsWithDeployKey(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "AssociateTagsWithDeployKey", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).AssociateTagsWithDeployKey), arg0, arg1)
}

// DeleteDevices mocks base method.
func (m *MockFleetMgrEdgeServiceServer) DeleteDevices(arg0 context.Context, arg1 *fmpb.DeleteDevicesRequest) (*fmpb.DeleteDevicesResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeleteDevices", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.DeleteDevicesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeleteDevices indicates an expected call of DeleteDevices.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) DeleteDevices(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteDevices", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).DeleteDevices), arg0, arg1)
}

// DeleteTag mocks base method.
func (m *MockFleetMgrEdgeServiceServer) DeleteTag(arg0 context.Context, arg1 *fmpb.DeleteTagRequest) (*fmpb.DeleteTagResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "DeleteTag", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.DeleteTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// DeleteTag indicates an expected call of DeleteTag.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) DeleteTag(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "DeleteTag", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).DeleteTag), arg0, arg1)
}

// GetDevice mocks base method.
func (m *MockFleetMgrEdgeServiceServer) GetDevice(arg0 context.Context, arg1 *fmpb.GetDeviceRequest) (*fmpb.GetDeviceResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetDevice", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.GetDeviceResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetDevice indicates an expected call of GetDevice.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) GetDevice(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetDevice", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).GetDevice), arg0, arg1)
}

// GetTags mocks base method.
func (m *MockFleetMgrEdgeServiceServer) GetTags(arg0 context.Context, arg1 *fmpb.GetTagsRequest) (*fmpb.GetTagsResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "GetTags", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.GetTagsResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// GetTags indicates an expected call of GetTags.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) GetTags(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "GetTags", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).GetTags), arg0, arg1)
}

// ListDevices mocks base method.
func (m *MockFleetMgrEdgeServiceServer) ListDevices(arg0 context.Context, arg1 *fmpb.ListDevicesRequest) (*fmpb.ListDevicesResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ListDevices", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.ListDevicesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListDevices indicates an expected call of ListDevices.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) ListDevices(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListDevices", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).ListDevices), arg0, arg1)
}

// ListTagsAssociatedWithDeployKey mocks base method.
func (m *MockFleetMgrEdgeServiceServer) ListTagsAssociatedWithDeployKey(arg0 context.Context, arg1 *fmpb.ListTagsAssociatedWithDeployKeyRequest) (*fmpb.ListTagsAssociatedWithDeployKeyResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "ListTagsAssociatedWithDeployKey", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.ListTagsAssociatedWithDeployKeyResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// ListTagsAssociatedWithDeployKey indicates an expected call of ListTagsAssociatedWithDeployKey.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) ListTagsAssociatedWithDeployKey(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "ListTagsAssociatedWithDeployKey", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).ListTagsAssociatedWithDeployKey), arg0, arg1)
}

// Register mocks base method.
func (m *MockFleetMgrEdgeServiceServer) Register(arg0 context.Context, arg1 *fmpb.RegisterRequest) (*fmpb.RegisterResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "Register", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.RegisterResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// Register indicates an expected call of Register.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) Register(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "Register", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).Register), arg0, arg1)
}

// SetDeviceCapabilities mocks base method.
func (m *MockFleetMgrEdgeServiceServer) SetDeviceCapabilities(arg0 context.Context, arg1 *fmpb.SetDeviceCapabilitiesRequest) (*fmpb.SetDeviceCapabilitiesResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "SetDeviceCapabilities", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.SetDeviceCapabilitiesResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// SetDeviceCapabilities indicates an expected call of SetDeviceCapabilities.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) SetDeviceCapabilities(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "SetDeviceCapabilities", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).SetDeviceCapabilities), arg0, arg1)
}

// UpdateDevice mocks base method.
func (m *MockFleetMgrEdgeServiceServer) UpdateDevice(arg0 context.Context, arg1 *fmpb.UpdateDeviceRequest) (*fmpb.UpdateDeviceResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "UpdateDevice", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.UpdateDeviceResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpdateDevice indicates an expected call of UpdateDevice.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) UpdateDevice(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpdateDevice", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).UpdateDevice), arg0, arg1)
}

// UpdateStatus mocks base method.
func (m *MockFleetMgrEdgeServiceServer) UpdateStatus(arg0 context.Context, arg1 *fmpb.UpdateStatusRequest) (*fmpb.UpdateStatusResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "UpdateStatus", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.UpdateStatusResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpdateStatus indicates an expected call of UpdateStatus.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) UpdateStatus(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpdateStatus", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).UpdateStatus), arg0, arg1)
}

// UpsertTag mocks base method.
func (m *MockFleetMgrEdgeServiceServer) UpsertTag(arg0 context.Context, arg1 *fmpb.UpsertTagRequest) (*fmpb.UpsertTagResponse, error) {
	m.ctrl.T.Helper()
	ret := m.ctrl.Call(m, "UpsertTag", arg0, arg1)
	ret0, _ := ret[0].(*fmpb.UpsertTagResponse)
	ret1, _ := ret[1].(error)
	return ret0, ret1
}

// UpsertTag indicates an expected call of UpsertTag.
func (mr *MockFleetMgrEdgeServiceServerMockRecorder) UpsertTag(arg0, arg1 any) *gomock.Call {
	mr.mock.ctrl.T.Helper()
	return mr.mock.ctrl.RecordCallWithMethodType(mr.mock, "UpsertTag", reflect.TypeOf((*MockFleetMgrEdgeServiceServer)(nil).UpsertTag), arg0, arg1)
}
