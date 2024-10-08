// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/calculators/video/box_tracker_calculator.proto

package video

import (
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
	framework "github.com/google/mediapipe/mediapipe/framework"
	tracking "github.com/google/mediapipe/mediapipe/util/tracking"
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

type BoxTrackerCalculatorOptions struct {
	TrackerOptions              *tracking.BoxTrackerOptions `protobuf:"bytes,1,opt,name=tracker_options,json=trackerOptions" json:"tracker_options,omitempty"`
	InitialPosition             *tracking.TimedBoxProtoList `protobuf:"bytes,2,opt,name=initial_position,json=initialPosition" json:"initial_position,omitempty"`
	VisualizeTrackingData       *bool                       `protobuf:"varint,3,opt,name=visualize_tracking_data,json=visualizeTrackingData,def=0" json:"visualize_tracking_data,omitempty"`
	VisualizeState              *bool                       `protobuf:"varint,4,opt,name=visualize_state,json=visualizeState,def=0" json:"visualize_state,omitempty"`
	VisualizeInternalState      *bool                       `protobuf:"varint,5,opt,name=visualize_internal_state,json=visualizeInternalState,def=0" json:"visualize_internal_state,omitempty"`
	StreamingTrackDataCacheSize *int32                      `protobuf:"varint,6,opt,name=streaming_track_data_cache_size,json=streamingTrackDataCacheSize,def=0" json:"streaming_track_data_cache_size,omitempty"`
	StartPosTransitionFrames    *int32                      `protobuf:"varint,7,opt,name=start_pos_transition_frames,json=startPosTransitionFrames,def=0" json:"start_pos_transition_frames,omitempty"`
}

func (m *BoxTrackerCalculatorOptions) Reset()      { *m = BoxTrackerCalculatorOptions{} }
func (*BoxTrackerCalculatorOptions) ProtoMessage() {}
func (*BoxTrackerCalculatorOptions) Descriptor() ([]byte, []int) {
	return fileDescriptor_fc3ae9b182a26166, []int{0}
}
func (m *BoxTrackerCalculatorOptions) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *BoxTrackerCalculatorOptions) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_BoxTrackerCalculatorOptions.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *BoxTrackerCalculatorOptions) XXX_Merge(src proto.Message) {
	xxx_messageInfo_BoxTrackerCalculatorOptions.Merge(m, src)
}
func (m *BoxTrackerCalculatorOptions) XXX_Size() int {
	return m.Size()
}
func (m *BoxTrackerCalculatorOptions) XXX_DiscardUnknown() {
	xxx_messageInfo_BoxTrackerCalculatorOptions.DiscardUnknown(m)
}

var xxx_messageInfo_BoxTrackerCalculatorOptions proto.InternalMessageInfo

const Default_BoxTrackerCalculatorOptions_VisualizeTrackingData bool = false
const Default_BoxTrackerCalculatorOptions_VisualizeState bool = false
const Default_BoxTrackerCalculatorOptions_VisualizeInternalState bool = false
const Default_BoxTrackerCalculatorOptions_StreamingTrackDataCacheSize int32 = 0
const Default_BoxTrackerCalculatorOptions_StartPosTransitionFrames int32 = 0

func (m *BoxTrackerCalculatorOptions) GetTrackerOptions() *tracking.BoxTrackerOptions {
	if m != nil {
		return m.TrackerOptions
	}
	return nil
}

func (m *BoxTrackerCalculatorOptions) GetInitialPosition() *tracking.TimedBoxProtoList {
	if m != nil {
		return m.InitialPosition
	}
	return nil
}

func (m *BoxTrackerCalculatorOptions) GetVisualizeTrackingData() bool {
	if m != nil && m.VisualizeTrackingData != nil {
		return *m.VisualizeTrackingData
	}
	return Default_BoxTrackerCalculatorOptions_VisualizeTrackingData
}

func (m *BoxTrackerCalculatorOptions) GetVisualizeState() bool {
	if m != nil && m.VisualizeState != nil {
		return *m.VisualizeState
	}
	return Default_BoxTrackerCalculatorOptions_VisualizeState
}

func (m *BoxTrackerCalculatorOptions) GetVisualizeInternalState() bool {
	if m != nil && m.VisualizeInternalState != nil {
		return *m.VisualizeInternalState
	}
	return Default_BoxTrackerCalculatorOptions_VisualizeInternalState
}

func (m *BoxTrackerCalculatorOptions) GetStreamingTrackDataCacheSize() int32 {
	if m != nil && m.StreamingTrackDataCacheSize != nil {
		return *m.StreamingTrackDataCacheSize
	}
	return Default_BoxTrackerCalculatorOptions_StreamingTrackDataCacheSize
}

func (m *BoxTrackerCalculatorOptions) GetStartPosTransitionFrames() int32 {
	if m != nil && m.StartPosTransitionFrames != nil {
		return *m.StartPosTransitionFrames
	}
	return Default_BoxTrackerCalculatorOptions_StartPosTransitionFrames
}

var E_BoxTrackerCalculatorOptions_Ext = &proto.ExtensionDesc{
	ExtendedType:  (*framework.CalculatorOptions)(nil),
	ExtensionType: (*BoxTrackerCalculatorOptions)(nil),
	Field:         268767860,
	Name:          "mediapipe.BoxTrackerCalculatorOptions.ext",
	Tag:           "bytes,268767860,opt,name=ext",
	Filename:      "mediapipe/calculators/video/box_tracker_calculator.proto",
}

func init() {
	proto.RegisterExtension(E_BoxTrackerCalculatorOptions_Ext)
	proto.RegisterType((*BoxTrackerCalculatorOptions)(nil), "mediapipe.BoxTrackerCalculatorOptions")
}

func init() {
	proto.RegisterFile("mediapipe/calculators/video/box_tracker_calculator.proto", fileDescriptor_fc3ae9b182a26166)
}

var fileDescriptor_fc3ae9b182a26166 = []byte{
	// 483 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x93, 0xcf, 0x6e, 0xd3, 0x40,
	0x10, 0xc6, 0xbd, 0xb4, 0xe1, 0xcf, 0x22, 0x35, 0xc8, 0x12, 0xb0, 0x6a, 0xd0, 0x12, 0x21, 0x84,
	0xc2, 0x25, 0x46, 0xbd, 0x80, 0x2a, 0x21, 0x50, 0x0a, 0x54, 0x48, 0x48, 0x44, 0x6e, 0x4e, 0xbd,
	0x58, 0x5b, 0x67, 0x9b, 0x8e, 0x6a, 0x7b, 0xad, 0xdd, 0x49, 0x89, 0x72, 0xea, 0x23, 0x70, 0xe0,
	0x11, 0x38, 0xf0, 0x20, 0x1c, 0x38, 0xe6, 0xd8, 0x23, 0x71, 0x2e, 0x1c, 0x7b, 0xe0, 0x01, 0xd0,
	0xda, 0x8e, 0x1d, 0x08, 0xea, 0xd1, 0x33, 0xbf, 0xef, 0xdb, 0xcf, 0x33, 0xbb, 0xf4, 0x45, 0x2c,
	0x87, 0x20, 0x52, 0x48, 0xa5, 0x17, 0x8a, 0x28, 0x1c, 0x47, 0x02, 0x95, 0x36, 0xde, 0x19, 0x0c,
	0xa5, 0xf2, 0x8e, 0xd4, 0x24, 0x40, 0x2d, 0xc2, 0x53, 0xa9, 0x83, 0xba, 0xdb, 0x4d, 0xb5, 0x42,
	0xe5, 0xde, 0xaa, 0x94, 0xdb, 0x8f, 0x6b, 0x93, 0x63, 0x2d, 0x62, 0xf9, 0x49, 0xe9, 0x53, 0xef,
	0x5f, 0xc1, 0xf6, 0xd3, 0x9a, 0x1a, 0x23, 0x44, 0x5e, 0xee, 0x0c, 0xc9, 0x68, 0xf5, 0x98, 0x02,
	0x7d, 0xf4, 0x7d, 0x93, 0xb6, 0x7a, 0x6a, 0x32, 0x28, 0x8a, 0x7b, 0x95, 0xd3, 0xc7, 0x14, 0x41,
	0x25, 0xc6, 0x7d, 0x4b, 0x9b, 0xcb, 0x5c, 0xaa, 0x28, 0x31, 0xd2, 0x26, 0x9d, 0xdb, 0x3b, 0x0f,
	0xba, 0xd5, 0x21, 0xdd, 0xda, 0xa0, 0x94, 0xf9, 0x5b, 0xf8, 0xd7, 0xb7, 0xbb, 0x4f, 0xef, 0x40,
	0x02, 0x08, 0x22, 0x0a, 0x52, 0x65, 0xc0, 0x16, 0xd9, 0xb5, 0x35, 0x9f, 0x01, 0xc4, 0x72, 0xd8,
	0x53, 0x93, 0xbe, 0x8d, 0xf6, 0x01, 0x0c, 0xfa, 0xcd, 0x52, 0xd5, 0x2f, 0x45, 0xee, 0x4b, 0x7a,
	0xff, 0x0c, 0xcc, 0x58, 0x44, 0x30, 0x95, 0xc1, 0xf2, 0xbf, 0x82, 0xa1, 0x40, 0xc1, 0x36, 0xda,
	0xa4, 0x73, 0x73, 0xb7, 0x71, 0x2c, 0x22, 0x23, 0xfd, 0xbb, 0x15, 0x35, 0x28, 0xa1, 0x37, 0x02,
	0x85, 0xdb, 0xa5, 0xcd, 0x5a, 0x6e, 0x50, 0xa0, 0x64, 0x9b, 0xab, 0xb2, 0xad, 0xaa, 0x7b, 0x60,
	0x9b, 0xee, 0x2b, 0xca, 0x6a, 0x1e, 0x12, 0x94, 0x3a, 0x11, 0x51, 0x29, 0x6c, 0xac, 0x0a, 0xef,
	0x55, 0xd8, 0xfb, 0x92, 0x2a, 0x0c, 0xf6, 0xe9, 0x43, 0x83, 0x5a, 0x8a, 0xd8, 0xc6, 0xcc, 0xf3,
	0xe6, 0x61, 0x83, 0x50, 0x84, 0x27, 0x32, 0x30, 0x30, 0x95, 0xec, 0x7a, 0x9b, 0x74, 0x1a, 0xbb,
	0xe4, 0x99, 0xdf, 0xaa, 0xc8, 0x3c, 0xb3, 0x0d, 0xbc, 0x67, 0xb1, 0x03, 0x98, 0x4a, 0xf7, 0x35,
	0x6d, 0x19, 0x14, 0x1a, 0xed, 0xfc, 0xac, 0x51, 0x52, 0x0c, 0x24, 0xc8, 0xaf, 0x81, 0x61, 0x37,
	0x96, 0x26, 0x2c, 0xa7, 0xfa, 0xca, 0x0c, 0x2a, 0xe6, 0x5d, 0x8e, 0xec, 0x1c, 0xd2, 0x0d, 0x39,
	0x41, 0x77, 0x75, 0xe0, 0x6b, 0xfb, 0x66, 0xbf, 0xbf, 0x7e, 0x39, 0x2f, 0x16, 0xfc, 0xe4, 0xbf,
	0x0b, 0x5e, 0x53, 0xf8, 0xd6, 0xb4, 0x17, 0xcf, 0xe6, 0xdc, 0xb9, 0x98, 0x73, 0xe7, 0x72, 0xce,
	0xc9, 0x79, 0xc6, 0xc9, 0xb7, 0x8c, 0x93, 0x1f, 0x19, 0x27, 0xb3, 0x8c, 0x93, 0x9f, 0x19, 0x27,
	0xbf, 0x32, 0xee, 0x5c, 0x66, 0x9c, 0x7c, 0x5e, 0x70, 0x67, 0xb6, 0xe0, 0xce, 0xc5, 0x82, 0x3b,
	0x87, 0xcf, 0x47, 0x80, 0x27, 0xe3, 0xa3, 0x6e, 0xa8, 0x62, 0x6f, 0xa4, 0xd4, 0x28, 0x92, 0x5e,
	0x7d, 0x73, 0xaf, 0x78, 0x2e, 0x7f, 0x02, 0x00, 0x00, 0xff, 0xff, 0x81, 0x9a, 0xf2, 0x90, 0x4c,
	0x03, 0x00, 0x00,
}

func (this *BoxTrackerCalculatorOptions) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*BoxTrackerCalculatorOptions)
	if !ok {
		that2, ok := that.(BoxTrackerCalculatorOptions)
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
	if !this.TrackerOptions.Equal(that1.TrackerOptions) {
		return false
	}
	if !this.InitialPosition.Equal(that1.InitialPosition) {
		return false
	}
	if this.VisualizeTrackingData != nil && that1.VisualizeTrackingData != nil {
		if *this.VisualizeTrackingData != *that1.VisualizeTrackingData {
			return false
		}
	} else if this.VisualizeTrackingData != nil {
		return false
	} else if that1.VisualizeTrackingData != nil {
		return false
	}
	if this.VisualizeState != nil && that1.VisualizeState != nil {
		if *this.VisualizeState != *that1.VisualizeState {
			return false
		}
	} else if this.VisualizeState != nil {
		return false
	} else if that1.VisualizeState != nil {
		return false
	}
	if this.VisualizeInternalState != nil && that1.VisualizeInternalState != nil {
		if *this.VisualizeInternalState != *that1.VisualizeInternalState {
			return false
		}
	} else if this.VisualizeInternalState != nil {
		return false
	} else if that1.VisualizeInternalState != nil {
		return false
	}
	if this.StreamingTrackDataCacheSize != nil && that1.StreamingTrackDataCacheSize != nil {
		if *this.StreamingTrackDataCacheSize != *that1.StreamingTrackDataCacheSize {
			return false
		}
	} else if this.StreamingTrackDataCacheSize != nil {
		return false
	} else if that1.StreamingTrackDataCacheSize != nil {
		return false
	}
	if this.StartPosTransitionFrames != nil && that1.StartPosTransitionFrames != nil {
		if *this.StartPosTransitionFrames != *that1.StartPosTransitionFrames {
			return false
		}
	} else if this.StartPosTransitionFrames != nil {
		return false
	} else if that1.StartPosTransitionFrames != nil {
		return false
	}
	return true
}
func (this *BoxTrackerCalculatorOptions) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 11)
	s = append(s, "&video.BoxTrackerCalculatorOptions{")
	if this.TrackerOptions != nil {
		s = append(s, "TrackerOptions: "+fmt.Sprintf("%#v", this.TrackerOptions)+",\n")
	}
	if this.InitialPosition != nil {
		s = append(s, "InitialPosition: "+fmt.Sprintf("%#v", this.InitialPosition)+",\n")
	}
	if this.VisualizeTrackingData != nil {
		s = append(s, "VisualizeTrackingData: "+valueToGoStringBoxTrackerCalculator(this.VisualizeTrackingData, "bool")+",\n")
	}
	if this.VisualizeState != nil {
		s = append(s, "VisualizeState: "+valueToGoStringBoxTrackerCalculator(this.VisualizeState, "bool")+",\n")
	}
	if this.VisualizeInternalState != nil {
		s = append(s, "VisualizeInternalState: "+valueToGoStringBoxTrackerCalculator(this.VisualizeInternalState, "bool")+",\n")
	}
	if this.StreamingTrackDataCacheSize != nil {
		s = append(s, "StreamingTrackDataCacheSize: "+valueToGoStringBoxTrackerCalculator(this.StreamingTrackDataCacheSize, "int32")+",\n")
	}
	if this.StartPosTransitionFrames != nil {
		s = append(s, "StartPosTransitionFrames: "+valueToGoStringBoxTrackerCalculator(this.StartPosTransitionFrames, "int32")+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringBoxTrackerCalculator(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *BoxTrackerCalculatorOptions) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *BoxTrackerCalculatorOptions) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *BoxTrackerCalculatorOptions) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if m.StartPosTransitionFrames != nil {
		i = encodeVarintBoxTrackerCalculator(dAtA, i, uint64(*m.StartPosTransitionFrames))
		i--
		dAtA[i] = 0x38
	}
	if m.StreamingTrackDataCacheSize != nil {
		i = encodeVarintBoxTrackerCalculator(dAtA, i, uint64(*m.StreamingTrackDataCacheSize))
		i--
		dAtA[i] = 0x30
	}
	if m.VisualizeInternalState != nil {
		i--
		if *m.VisualizeInternalState {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x28
	}
	if m.VisualizeState != nil {
		i--
		if *m.VisualizeState {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x20
	}
	if m.VisualizeTrackingData != nil {
		i--
		if *m.VisualizeTrackingData {
			dAtA[i] = 1
		} else {
			dAtA[i] = 0
		}
		i--
		dAtA[i] = 0x18
	}
	if m.InitialPosition != nil {
		{
			size, err := m.InitialPosition.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintBoxTrackerCalculator(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x12
	}
	if m.TrackerOptions != nil {
		{
			size, err := m.TrackerOptions.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintBoxTrackerCalculator(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0xa
	}
	return len(dAtA) - i, nil
}

func encodeVarintBoxTrackerCalculator(dAtA []byte, offset int, v uint64) int {
	offset -= sovBoxTrackerCalculator(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *BoxTrackerCalculatorOptions) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if m.TrackerOptions != nil {
		l = m.TrackerOptions.Size()
		n += 1 + l + sovBoxTrackerCalculator(uint64(l))
	}
	if m.InitialPosition != nil {
		l = m.InitialPosition.Size()
		n += 1 + l + sovBoxTrackerCalculator(uint64(l))
	}
	if m.VisualizeTrackingData != nil {
		n += 2
	}
	if m.VisualizeState != nil {
		n += 2
	}
	if m.VisualizeInternalState != nil {
		n += 2
	}
	if m.StreamingTrackDataCacheSize != nil {
		n += 1 + sovBoxTrackerCalculator(uint64(*m.StreamingTrackDataCacheSize))
	}
	if m.StartPosTransitionFrames != nil {
		n += 1 + sovBoxTrackerCalculator(uint64(*m.StartPosTransitionFrames))
	}
	return n
}

func sovBoxTrackerCalculator(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozBoxTrackerCalculator(x uint64) (n int) {
	return sovBoxTrackerCalculator(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *BoxTrackerCalculatorOptions) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&BoxTrackerCalculatorOptions{`,
		`TrackerOptions:` + strings.Replace(fmt.Sprintf("%v", this.TrackerOptions), "BoxTrackerOptions", "tracking.BoxTrackerOptions", 1) + `,`,
		`InitialPosition:` + strings.Replace(fmt.Sprintf("%v", this.InitialPosition), "TimedBoxProtoList", "tracking.TimedBoxProtoList", 1) + `,`,
		`VisualizeTrackingData:` + valueToStringBoxTrackerCalculator(this.VisualizeTrackingData) + `,`,
		`VisualizeState:` + valueToStringBoxTrackerCalculator(this.VisualizeState) + `,`,
		`VisualizeInternalState:` + valueToStringBoxTrackerCalculator(this.VisualizeInternalState) + `,`,
		`StreamingTrackDataCacheSize:` + valueToStringBoxTrackerCalculator(this.StreamingTrackDataCacheSize) + `,`,
		`StartPosTransitionFrames:` + valueToStringBoxTrackerCalculator(this.StartPosTransitionFrames) + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringBoxTrackerCalculator(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *BoxTrackerCalculatorOptions) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowBoxTrackerCalculator
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
			return fmt.Errorf("proto: BoxTrackerCalculatorOptions: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: BoxTrackerCalculatorOptions: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field TrackerOptions", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowBoxTrackerCalculator
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
				return ErrInvalidLengthBoxTrackerCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthBoxTrackerCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.TrackerOptions == nil {
				m.TrackerOptions = &tracking.BoxTrackerOptions{}
			}
			if err := m.TrackerOptions.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 2:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field InitialPosition", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowBoxTrackerCalculator
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
				return ErrInvalidLengthBoxTrackerCalculator
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthBoxTrackerCalculator
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.InitialPosition == nil {
				m.InitialPosition = &tracking.TimedBoxProtoList{}
			}
			if err := m.InitialPosition.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 3:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field VisualizeTrackingData", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowBoxTrackerCalculator
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
			b := bool(v != 0)
			m.VisualizeTrackingData = &b
		case 4:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field VisualizeState", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowBoxTrackerCalculator
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
			b := bool(v != 0)
			m.VisualizeState = &b
		case 5:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field VisualizeInternalState", wireType)
			}
			var v int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowBoxTrackerCalculator
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
			b := bool(v != 0)
			m.VisualizeInternalState = &b
		case 6:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field StreamingTrackDataCacheSize", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowBoxTrackerCalculator
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
			m.StreamingTrackDataCacheSize = &v
		case 7:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field StartPosTransitionFrames", wireType)
			}
			var v int32
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowBoxTrackerCalculator
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
			m.StartPosTransitionFrames = &v
		default:
			iNdEx = preIndex
			skippy, err := skipBoxTrackerCalculator(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthBoxTrackerCalculator
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
func skipBoxTrackerCalculator(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowBoxTrackerCalculator
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
					return 0, ErrIntOverflowBoxTrackerCalculator
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
					return 0, ErrIntOverflowBoxTrackerCalculator
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
				return 0, ErrInvalidLengthBoxTrackerCalculator
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupBoxTrackerCalculator
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthBoxTrackerCalculator
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthBoxTrackerCalculator        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowBoxTrackerCalculator          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupBoxTrackerCalculator = fmt.Errorf("proto: unexpected end of group")
)
