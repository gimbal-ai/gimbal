// Code generated by protoc-gen-gogo. DO NOT EDIT.
// source: mediapipe/framework/formats/detection.proto

package formats

import (
	encoding_binary "encoding/binary"
	fmt "fmt"
	proto "github.com/gogo/protobuf/proto"
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

type Detection struct {
	Label                []string                         `protobuf:"bytes,1,rep,name=label" json:"label,omitempty"`
	LabelId              []int32                          `protobuf:"varint,2,rep,packed,name=label_id,json=labelId" json:"label_id,omitempty"`
	Score                []float32                        `protobuf:"fixed32,3,rep,packed,name=score" json:"score,omitempty"`
	LocationData         *LocationData                    `protobuf:"bytes,4,opt,name=location_data,json=locationData" json:"location_data,omitempty"`
	FeatureTag           string                           `protobuf:"bytes,5,opt,name=feature_tag,json=featureTag" json:"feature_tag"`
	TrackId              string                           `protobuf:"bytes,6,opt,name=track_id,json=trackId" json:"track_id"`
	DetectionId          int64                            `protobuf:"varint,7,opt,name=detection_id,json=detectionId" json:"detection_id"`
	AssociatedDetections []*Detection_AssociatedDetection `protobuf:"bytes,8,rep,name=associated_detections,json=associatedDetections" json:"associated_detections,omitempty"`
	DisplayName          []string                         `protobuf:"bytes,9,rep,name=display_name,json=displayName" json:"display_name,omitempty"`
	TimestampUsec        int64                            `protobuf:"varint,10,opt,name=timestamp_usec,json=timestampUsec" json:"timestamp_usec"`
}

func (m *Detection) Reset()      { *m = Detection{} }
func (*Detection) ProtoMessage() {}
func (*Detection) Descriptor() ([]byte, []int) {
	return fileDescriptor_bf734d49e56f65c7, []int{0}
}
func (m *Detection) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Detection) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Detection.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Detection) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Detection.Merge(m, src)
}
func (m *Detection) XXX_Size() int {
	return m.Size()
}
func (m *Detection) XXX_DiscardUnknown() {
	xxx_messageInfo_Detection.DiscardUnknown(m)
}

var xxx_messageInfo_Detection proto.InternalMessageInfo

func (m *Detection) GetLabel() []string {
	if m != nil {
		return m.Label
	}
	return nil
}

func (m *Detection) GetLabelId() []int32 {
	if m != nil {
		return m.LabelId
	}
	return nil
}

func (m *Detection) GetScore() []float32 {
	if m != nil {
		return m.Score
	}
	return nil
}

func (m *Detection) GetLocationData() *LocationData {
	if m != nil {
		return m.LocationData
	}
	return nil
}

func (m *Detection) GetFeatureTag() string {
	if m != nil {
		return m.FeatureTag
	}
	return ""
}

func (m *Detection) GetTrackId() string {
	if m != nil {
		return m.TrackId
	}
	return ""
}

func (m *Detection) GetDetectionId() int64 {
	if m != nil {
		return m.DetectionId
	}
	return 0
}

func (m *Detection) GetAssociatedDetections() []*Detection_AssociatedDetection {
	if m != nil {
		return m.AssociatedDetections
	}
	return nil
}

func (m *Detection) GetDisplayName() []string {
	if m != nil {
		return m.DisplayName
	}
	return nil
}

func (m *Detection) GetTimestampUsec() int64 {
	if m != nil {
		return m.TimestampUsec
	}
	return 0
}

type Detection_AssociatedDetection struct {
	Id         int32   `protobuf:"varint,1,opt,name=id" json:"id"`
	Confidence float32 `protobuf:"fixed32,2,opt,name=confidence" json:"confidence"`
}

func (m *Detection_AssociatedDetection) Reset()      { *m = Detection_AssociatedDetection{} }
func (*Detection_AssociatedDetection) ProtoMessage() {}
func (*Detection_AssociatedDetection) Descriptor() ([]byte, []int) {
	return fileDescriptor_bf734d49e56f65c7, []int{0, 0}
}
func (m *Detection_AssociatedDetection) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *Detection_AssociatedDetection) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_Detection_AssociatedDetection.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *Detection_AssociatedDetection) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Detection_AssociatedDetection.Merge(m, src)
}
func (m *Detection_AssociatedDetection) XXX_Size() int {
	return m.Size()
}
func (m *Detection_AssociatedDetection) XXX_DiscardUnknown() {
	xxx_messageInfo_Detection_AssociatedDetection.DiscardUnknown(m)
}

var xxx_messageInfo_Detection_AssociatedDetection proto.InternalMessageInfo

func (m *Detection_AssociatedDetection) GetId() int32 {
	if m != nil {
		return m.Id
	}
	return 0
}

func (m *Detection_AssociatedDetection) GetConfidence() float32 {
	if m != nil {
		return m.Confidence
	}
	return 0
}

type DetectionList struct {
	Detection []*Detection `protobuf:"bytes,1,rep,name=detection" json:"detection,omitempty"`
}

func (m *DetectionList) Reset()      { *m = DetectionList{} }
func (*DetectionList) ProtoMessage() {}
func (*DetectionList) Descriptor() ([]byte, []int) {
	return fileDescriptor_bf734d49e56f65c7, []int{1}
}
func (m *DetectionList) XXX_Unmarshal(b []byte) error {
	return m.Unmarshal(b)
}
func (m *DetectionList) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	if deterministic {
		return xxx_messageInfo_DetectionList.Marshal(b, m, deterministic)
	} else {
		b = b[:cap(b)]
		n, err := m.MarshalToSizedBuffer(b)
		if err != nil {
			return nil, err
		}
		return b[:n], nil
	}
}
func (m *DetectionList) XXX_Merge(src proto.Message) {
	xxx_messageInfo_DetectionList.Merge(m, src)
}
func (m *DetectionList) XXX_Size() int {
	return m.Size()
}
func (m *DetectionList) XXX_DiscardUnknown() {
	xxx_messageInfo_DetectionList.DiscardUnknown(m)
}

var xxx_messageInfo_DetectionList proto.InternalMessageInfo

func (m *DetectionList) GetDetection() []*Detection {
	if m != nil {
		return m.Detection
	}
	return nil
}

func init() {
	proto.RegisterType((*Detection)(nil), "mediapipe.Detection")
	proto.RegisterType((*Detection_AssociatedDetection)(nil), "mediapipe.Detection.AssociatedDetection")
	proto.RegisterType((*DetectionList)(nil), "mediapipe.DetectionList")
}

func init() {
	proto.RegisterFile("mediapipe/framework/formats/detection.proto", fileDescriptor_bf734d49e56f65c7)
}

var fileDescriptor_bf734d49e56f65c7 = []byte{
	// 507 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x7c, 0x53, 0xcf, 0x8b, 0xd3, 0x4e,
	0x14, 0xcf, 0x24, 0xed, 0xb7, 0xcd, 0xa4, 0x5d, 0xbe, 0x8c, 0x15, 0x87, 0x05, 0x67, 0x63, 0x51,
	0x0c, 0x2c, 0xa4, 0xd0, 0x8b, 0x17, 0x2f, 0xd6, 0xbd, 0x14, 0x16, 0xd1, 0xa0, 0x17, 0x41, 0xc2,
	0xec, 0xcc, 0xb4, 0x0e, 0x9b, 0x74, 0x4a, 0x66, 0x8a, 0x78, 0xf3, 0xee, 0xc5, 0x7f, 0xc1, 0x9b,
	0x7f, 0xca, 0x1e, 0x7b, 0xdc, 0x93, 0xd8, 0xf4, 0xe2, 0x71, 0xff, 0x04, 0x69, 0x9a, 0x9d, 0x8d,
	0xb8, 0xec, 0x6d, 0xde, 0xe7, 0xc7, 0xbc, 0xc7, 0x87, 0xf7, 0xe0, 0x71, 0x2e, 0xb8, 0xa4, 0x4b,
	0xb9, 0x14, 0xa3, 0x59, 0x41, 0x73, 0xf1, 0x49, 0x15, 0xe7, 0xa3, 0x99, 0x2a, 0x72, 0x6a, 0xf4,
	0x88, 0x0b, 0x23, 0x98, 0x91, 0x6a, 0x11, 0x2f, 0x0b, 0x65, 0x14, 0xf2, 0xad, 0xf8, 0x70, 0x74,
	0x97, 0x2f, 0x53, 0x8c, 0xee, 0x6c, 0x29, 0xa7, 0x86, 0xee, 0xbd, 0xc3, 0xaf, 0x2d, 0xe8, 0x9f,
	0x5c, 0xff, 0x87, 0x06, 0xb0, 0x9d, 0xd1, 0x33, 0x91, 0x61, 0x10, 0x7a, 0x91, 0x9f, 0xec, 0x0b,
	0xf4, 0x10, 0x76, 0xab, 0x47, 0x2a, 0x39, 0x76, 0x43, 0x2f, 0x6a, 0x4f, 0xdc, 0xff, 0x41, 0xd2,
	0xa9, 0xb0, 0x29, 0x47, 0x18, 0xb6, 0x35, 0x53, 0x85, 0xc0, 0x5e, 0xe8, 0x45, 0x6e, 0xc5, 0xed,
	0x01, 0xf4, 0x1c, 0xf6, 0xff, 0xea, 0x89, 0x5b, 0x21, 0x88, 0x82, 0xf1, 0x83, 0xd8, 0x4e, 0x19,
	0x9f, 0xd6, 0xfc, 0x09, 0x35, 0x34, 0xe9, 0x65, 0x8d, 0x0a, 0x3d, 0x81, 0xc1, 0x4c, 0x50, 0xb3,
	0x2a, 0x44, 0x6a, 0xe8, 0x1c, 0xb7, 0x43, 0x10, 0xf9, 0x93, 0xd6, 0xc5, 0xcf, 0x23, 0x27, 0x81,
	0x35, 0xf1, 0x96, 0xce, 0xd1, 0x11, 0xec, 0x9a, 0x82, 0xb2, 0xf3, 0xdd, 0x74, 0xff, 0x35, 0x34,
	0x9d, 0x0a, 0x9d, 0x72, 0xf4, 0x14, 0xf6, 0x6c, 0x62, 0x3b, 0x51, 0x27, 0x04, 0x91, 0x57, 0x8b,
	0x02, 0xcb, 0x4c, 0x39, 0xfa, 0x00, 0xef, 0x53, 0xad, 0x15, 0x93, 0xd4, 0x08, 0x9e, 0x5a, 0x46,
	0xe3, 0x6e, 0xe8, 0x45, 0xc1, 0x38, 0x6a, 0x8c, 0x6d, 0x23, 0x8b, 0x5f, 0x58, 0x87, 0xc5, 0x92,
	0x01, 0xfd, 0x17, 0xd4, 0xe8, 0x11, 0xec, 0x71, 0xa9, 0x97, 0x19, 0xfd, 0x9c, 0x2e, 0x68, 0x2e,
	0xb0, 0x5f, 0x65, 0x1c, 0xd4, 0xd8, 0x2b, 0x9a, 0x0b, 0x74, 0x0c, 0x0f, 0x8c, 0xcc, 0x85, 0x36,
	0x34, 0x5f, 0xa6, 0x2b, 0x2d, 0x18, 0x86, 0x8d, 0x61, 0xfb, 0x96, 0x7b, 0xa7, 0x05, 0x3b, 0x7c,
	0x03, 0xef, 0xdd, 0xd2, 0x1c, 0x0d, 0xa0, 0x2b, 0x39, 0x06, 0x21, 0x88, 0xda, 0xb5, 0xcf, 0x95,
	0x1c, 0x3d, 0x86, 0x90, 0xa9, 0xc5, 0x4c, 0x72, 0xb1, 0x60, 0x02, 0xbb, 0x21, 0x88, 0xdc, 0xeb,
	0x2c, 0x6f, 0xf0, 0xe1, 0x4b, 0xd8, 0xb7, 0x1f, 0x9d, 0x4a, 0x6d, 0xd0, 0x18, 0xfa, 0x36, 0x87,
	0x6a, 0x29, 0x82, 0xf1, 0xe0, 0xb6, 0x18, 0x92, 0x1b, 0xd9, 0xe4, 0x3b, 0x58, 0x6f, 0x88, 0x73,
	0xb9, 0x21, 0xce, 0xd5, 0x86, 0x80, 0x2f, 0x25, 0x01, 0x3f, 0x4a, 0x02, 0x2e, 0x4a, 0x02, 0xd6,
	0x25, 0x01, 0xbf, 0x4a, 0x02, 0x7e, 0x97, 0xc4, 0xb9, 0x2a, 0x09, 0xf8, 0xb6, 0x25, 0xce, 0x7a,
	0x4b, 0x9c, 0xcb, 0x2d, 0x71, 0xe0, 0x90, 0xa9, 0x3c, 0x9e, 0x2b, 0x35, 0xcf, 0x44, 0xa3, 0x43,
	0xbd, 0xbb, 0xfb, 0x6d, 0x9d, 0x1c, 0xd8, 0x86, 0xaf, 0x77, 0xf5, 0xfb, 0x67, 0x73, 0x69, 0x3e,
	0xae, 0xce, 0x62, 0xa6, 0xf2, 0xd1, 0xde, 0xda, 0x38, 0x80, 0x3b, 0x4e, 0xe1, 0x4f, 0x00, 0x00,
	0x00, 0xff, 0xff, 0x20, 0xab, 0x65, 0x2d, 0x60, 0x03, 0x00, 0x00,
}

func (this *Detection) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*Detection)
	if !ok {
		that2, ok := that.(Detection)
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
	if len(this.Label) != len(that1.Label) {
		return false
	}
	for i := range this.Label {
		if this.Label[i] != that1.Label[i] {
			return false
		}
	}
	if len(this.LabelId) != len(that1.LabelId) {
		return false
	}
	for i := range this.LabelId {
		if this.LabelId[i] != that1.LabelId[i] {
			return false
		}
	}
	if len(this.Score) != len(that1.Score) {
		return false
	}
	for i := range this.Score {
		if this.Score[i] != that1.Score[i] {
			return false
		}
	}
	if !this.LocationData.Equal(that1.LocationData) {
		return false
	}
	if this.FeatureTag != that1.FeatureTag {
		return false
	}
	if this.TrackId != that1.TrackId {
		return false
	}
	if this.DetectionId != that1.DetectionId {
		return false
	}
	if len(this.AssociatedDetections) != len(that1.AssociatedDetections) {
		return false
	}
	for i := range this.AssociatedDetections {
		if !this.AssociatedDetections[i].Equal(that1.AssociatedDetections[i]) {
			return false
		}
	}
	if len(this.DisplayName) != len(that1.DisplayName) {
		return false
	}
	for i := range this.DisplayName {
		if this.DisplayName[i] != that1.DisplayName[i] {
			return false
		}
	}
	if this.TimestampUsec != that1.TimestampUsec {
		return false
	}
	return true
}
func (this *Detection_AssociatedDetection) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*Detection_AssociatedDetection)
	if !ok {
		that2, ok := that.(Detection_AssociatedDetection)
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
	if this.Id != that1.Id {
		return false
	}
	if this.Confidence != that1.Confidence {
		return false
	}
	return true
}
func (this *DetectionList) Equal(that interface{}) bool {
	if that == nil {
		return this == nil
	}

	that1, ok := that.(*DetectionList)
	if !ok {
		that2, ok := that.(DetectionList)
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
	if len(this.Detection) != len(that1.Detection) {
		return false
	}
	for i := range this.Detection {
		if !this.Detection[i].Equal(that1.Detection[i]) {
			return false
		}
	}
	return true
}
func (this *Detection) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 14)
	s = append(s, "&formats.Detection{")
	if this.Label != nil {
		s = append(s, "Label: "+fmt.Sprintf("%#v", this.Label)+",\n")
	}
	if this.LabelId != nil {
		s = append(s, "LabelId: "+fmt.Sprintf("%#v", this.LabelId)+",\n")
	}
	if this.Score != nil {
		s = append(s, "Score: "+fmt.Sprintf("%#v", this.Score)+",\n")
	}
	if this.LocationData != nil {
		s = append(s, "LocationData: "+fmt.Sprintf("%#v", this.LocationData)+",\n")
	}
	s = append(s, "FeatureTag: "+fmt.Sprintf("%#v", this.FeatureTag)+",\n")
	s = append(s, "TrackId: "+fmt.Sprintf("%#v", this.TrackId)+",\n")
	s = append(s, "DetectionId: "+fmt.Sprintf("%#v", this.DetectionId)+",\n")
	if this.AssociatedDetections != nil {
		s = append(s, "AssociatedDetections: "+fmt.Sprintf("%#v", this.AssociatedDetections)+",\n")
	}
	if this.DisplayName != nil {
		s = append(s, "DisplayName: "+fmt.Sprintf("%#v", this.DisplayName)+",\n")
	}
	s = append(s, "TimestampUsec: "+fmt.Sprintf("%#v", this.TimestampUsec)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *Detection_AssociatedDetection) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 6)
	s = append(s, "&formats.Detection_AssociatedDetection{")
	s = append(s, "Id: "+fmt.Sprintf("%#v", this.Id)+",\n")
	s = append(s, "Confidence: "+fmt.Sprintf("%#v", this.Confidence)+",\n")
	s = append(s, "}")
	return strings.Join(s, "")
}
func (this *DetectionList) GoString() string {
	if this == nil {
		return "nil"
	}
	s := make([]string, 0, 5)
	s = append(s, "&formats.DetectionList{")
	if this.Detection != nil {
		s = append(s, "Detection: "+fmt.Sprintf("%#v", this.Detection)+",\n")
	}
	s = append(s, "}")
	return strings.Join(s, "")
}
func valueToGoStringDetection(v interface{}, typ string) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("func(v %v) *%v { return &v } ( %#v )", typ, typ, pv)
}
func (m *Detection) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Detection) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Detection) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i = encodeVarintDetection(dAtA, i, uint64(m.TimestampUsec))
	i--
	dAtA[i] = 0x50
	if len(m.DisplayName) > 0 {
		for iNdEx := len(m.DisplayName) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.DisplayName[iNdEx])
			copy(dAtA[i:], m.DisplayName[iNdEx])
			i = encodeVarintDetection(dAtA, i, uint64(len(m.DisplayName[iNdEx])))
			i--
			dAtA[i] = 0x4a
		}
	}
	if len(m.AssociatedDetections) > 0 {
		for iNdEx := len(m.AssociatedDetections) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.AssociatedDetections[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintDetection(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0x42
		}
	}
	i = encodeVarintDetection(dAtA, i, uint64(m.DetectionId))
	i--
	dAtA[i] = 0x38
	i -= len(m.TrackId)
	copy(dAtA[i:], m.TrackId)
	i = encodeVarintDetection(dAtA, i, uint64(len(m.TrackId)))
	i--
	dAtA[i] = 0x32
	i -= len(m.FeatureTag)
	copy(dAtA[i:], m.FeatureTag)
	i = encodeVarintDetection(dAtA, i, uint64(len(m.FeatureTag)))
	i--
	dAtA[i] = 0x2a
	if m.LocationData != nil {
		{
			size, err := m.LocationData.MarshalToSizedBuffer(dAtA[:i])
			if err != nil {
				return 0, err
			}
			i -= size
			i = encodeVarintDetection(dAtA, i, uint64(size))
		}
		i--
		dAtA[i] = 0x22
	}
	if len(m.Score) > 0 {
		for iNdEx := len(m.Score) - 1; iNdEx >= 0; iNdEx-- {
			f2 := math.Float32bits(float32(m.Score[iNdEx]))
			i -= 4
			encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(f2))
		}
		i = encodeVarintDetection(dAtA, i, uint64(len(m.Score)*4))
		i--
		dAtA[i] = 0x1a
	}
	if len(m.LabelId) > 0 {
		dAtA4 := make([]byte, len(m.LabelId)*10)
		var j3 int
		for _, num1 := range m.LabelId {
			num := uint64(num1)
			for num >= 1<<7 {
				dAtA4[j3] = uint8(uint64(num)&0x7f | 0x80)
				num >>= 7
				j3++
			}
			dAtA4[j3] = uint8(num)
			j3++
		}
		i -= j3
		copy(dAtA[i:], dAtA4[:j3])
		i = encodeVarintDetection(dAtA, i, uint64(j3))
		i--
		dAtA[i] = 0x12
	}
	if len(m.Label) > 0 {
		for iNdEx := len(m.Label) - 1; iNdEx >= 0; iNdEx-- {
			i -= len(m.Label[iNdEx])
			copy(dAtA[i:], m.Label[iNdEx])
			i = encodeVarintDetection(dAtA, i, uint64(len(m.Label[iNdEx])))
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func (m *Detection_AssociatedDetection) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *Detection_AssociatedDetection) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *Detection_AssociatedDetection) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	i -= 4
	encoding_binary.LittleEndian.PutUint32(dAtA[i:], uint32(math.Float32bits(float32(m.Confidence))))
	i--
	dAtA[i] = 0x15
	i = encodeVarintDetection(dAtA, i, uint64(m.Id))
	i--
	dAtA[i] = 0x8
	return len(dAtA) - i, nil
}

func (m *DetectionList) Marshal() (dAtA []byte, err error) {
	size := m.Size()
	dAtA = make([]byte, size)
	n, err := m.MarshalToSizedBuffer(dAtA[:size])
	if err != nil {
		return nil, err
	}
	return dAtA[:n], nil
}

func (m *DetectionList) MarshalTo(dAtA []byte) (int, error) {
	size := m.Size()
	return m.MarshalToSizedBuffer(dAtA[:size])
}

func (m *DetectionList) MarshalToSizedBuffer(dAtA []byte) (int, error) {
	i := len(dAtA)
	_ = i
	var l int
	_ = l
	if len(m.Detection) > 0 {
		for iNdEx := len(m.Detection) - 1; iNdEx >= 0; iNdEx-- {
			{
				size, err := m.Detection[iNdEx].MarshalToSizedBuffer(dAtA[:i])
				if err != nil {
					return 0, err
				}
				i -= size
				i = encodeVarintDetection(dAtA, i, uint64(size))
			}
			i--
			dAtA[i] = 0xa
		}
	}
	return len(dAtA) - i, nil
}

func encodeVarintDetection(dAtA []byte, offset int, v uint64) int {
	offset -= sovDetection(v)
	base := offset
	for v >= 1<<7 {
		dAtA[offset] = uint8(v&0x7f | 0x80)
		v >>= 7
		offset++
	}
	dAtA[offset] = uint8(v)
	return base
}
func (m *Detection) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Label) > 0 {
		for _, s := range m.Label {
			l = len(s)
			n += 1 + l + sovDetection(uint64(l))
		}
	}
	if len(m.LabelId) > 0 {
		l = 0
		for _, e := range m.LabelId {
			l += sovDetection(uint64(e))
		}
		n += 1 + sovDetection(uint64(l)) + l
	}
	if len(m.Score) > 0 {
		n += 1 + sovDetection(uint64(len(m.Score)*4)) + len(m.Score)*4
	}
	if m.LocationData != nil {
		l = m.LocationData.Size()
		n += 1 + l + sovDetection(uint64(l))
	}
	l = len(m.FeatureTag)
	n += 1 + l + sovDetection(uint64(l))
	l = len(m.TrackId)
	n += 1 + l + sovDetection(uint64(l))
	n += 1 + sovDetection(uint64(m.DetectionId))
	if len(m.AssociatedDetections) > 0 {
		for _, e := range m.AssociatedDetections {
			l = e.Size()
			n += 1 + l + sovDetection(uint64(l))
		}
	}
	if len(m.DisplayName) > 0 {
		for _, s := range m.DisplayName {
			l = len(s)
			n += 1 + l + sovDetection(uint64(l))
		}
	}
	n += 1 + sovDetection(uint64(m.TimestampUsec))
	return n
}

func (m *Detection_AssociatedDetection) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	n += 1 + sovDetection(uint64(m.Id))
	n += 5
	return n
}

func (m *DetectionList) Size() (n int) {
	if m == nil {
		return 0
	}
	var l int
	_ = l
	if len(m.Detection) > 0 {
		for _, e := range m.Detection {
			l = e.Size()
			n += 1 + l + sovDetection(uint64(l))
		}
	}
	return n
}

func sovDetection(x uint64) (n int) {
	return (math_bits.Len64(x|1) + 6) / 7
}
func sozDetection(x uint64) (n int) {
	return sovDetection(uint64((x << 1) ^ uint64((int64(x) >> 63))))
}
func (this *Detection) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForAssociatedDetections := "[]*Detection_AssociatedDetection{"
	for _, f := range this.AssociatedDetections {
		repeatedStringForAssociatedDetections += strings.Replace(fmt.Sprintf("%v", f), "Detection_AssociatedDetection", "Detection_AssociatedDetection", 1) + ","
	}
	repeatedStringForAssociatedDetections += "}"
	s := strings.Join([]string{`&Detection{`,
		`Label:` + fmt.Sprintf("%v", this.Label) + `,`,
		`LabelId:` + fmt.Sprintf("%v", this.LabelId) + `,`,
		`Score:` + fmt.Sprintf("%v", this.Score) + `,`,
		`LocationData:` + strings.Replace(fmt.Sprintf("%v", this.LocationData), "LocationData", "LocationData", 1) + `,`,
		`FeatureTag:` + fmt.Sprintf("%v", this.FeatureTag) + `,`,
		`TrackId:` + fmt.Sprintf("%v", this.TrackId) + `,`,
		`DetectionId:` + fmt.Sprintf("%v", this.DetectionId) + `,`,
		`AssociatedDetections:` + repeatedStringForAssociatedDetections + `,`,
		`DisplayName:` + fmt.Sprintf("%v", this.DisplayName) + `,`,
		`TimestampUsec:` + fmt.Sprintf("%v", this.TimestampUsec) + `,`,
		`}`,
	}, "")
	return s
}
func (this *Detection_AssociatedDetection) String() string {
	if this == nil {
		return "nil"
	}
	s := strings.Join([]string{`&Detection_AssociatedDetection{`,
		`Id:` + fmt.Sprintf("%v", this.Id) + `,`,
		`Confidence:` + fmt.Sprintf("%v", this.Confidence) + `,`,
		`}`,
	}, "")
	return s
}
func (this *DetectionList) String() string {
	if this == nil {
		return "nil"
	}
	repeatedStringForDetection := "[]*Detection{"
	for _, f := range this.Detection {
		repeatedStringForDetection += strings.Replace(f.String(), "Detection", "Detection", 1) + ","
	}
	repeatedStringForDetection += "}"
	s := strings.Join([]string{`&DetectionList{`,
		`Detection:` + repeatedStringForDetection + `,`,
		`}`,
	}, "")
	return s
}
func valueToStringDetection(v interface{}) string {
	rv := reflect.ValueOf(v)
	if rv.IsNil() {
		return "nil"
	}
	pv := reflect.Indirect(rv).Interface()
	return fmt.Sprintf("*%v", pv)
}
func (m *Detection) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowDetection
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
			return fmt.Errorf("proto: Detection: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: Detection: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Label", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
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
				return ErrInvalidLengthDetection
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthDetection
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Label = append(m.Label, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		case 2:
			if wireType == 0 {
				var v int32
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDetection
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
				m.LabelId = append(m.LabelId, v)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDetection
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
					return ErrInvalidLengthDetection
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthDetection
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
				if elementCount != 0 && len(m.LabelId) == 0 {
					m.LabelId = make([]int32, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v int32
					for shift := uint(0); ; shift += 7 {
						if shift >= 64 {
							return ErrIntOverflowDetection
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
					m.LabelId = append(m.LabelId, v)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field LabelId", wireType)
			}
		case 3:
			if wireType == 5 {
				var v uint32
				if (iNdEx + 4) > l {
					return io.ErrUnexpectedEOF
				}
				v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
				iNdEx += 4
				v2 := float32(math.Float32frombits(v))
				m.Score = append(m.Score, v2)
			} else if wireType == 2 {
				var packedLen int
				for shift := uint(0); ; shift += 7 {
					if shift >= 64 {
						return ErrIntOverflowDetection
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
					return ErrInvalidLengthDetection
				}
				postIndex := iNdEx + packedLen
				if postIndex < 0 {
					return ErrInvalidLengthDetection
				}
				if postIndex > l {
					return io.ErrUnexpectedEOF
				}
				var elementCount int
				elementCount = packedLen / 4
				if elementCount != 0 && len(m.Score) == 0 {
					m.Score = make([]float32, 0, elementCount)
				}
				for iNdEx < postIndex {
					var v uint32
					if (iNdEx + 4) > l {
						return io.ErrUnexpectedEOF
					}
					v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
					iNdEx += 4
					v2 := float32(math.Float32frombits(v))
					m.Score = append(m.Score, v2)
				}
			} else {
				return fmt.Errorf("proto: wrong wireType = %d for field Score", wireType)
			}
		case 4:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field LocationData", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
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
				return ErrInvalidLengthDetection
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthDetection
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			if m.LocationData == nil {
				m.LocationData = &LocationData{}
			}
			if err := m.LocationData.Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 5:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field FeatureTag", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
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
				return ErrInvalidLengthDetection
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthDetection
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.FeatureTag = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 6:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field TrackId", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
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
				return ErrInvalidLengthDetection
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthDetection
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.TrackId = string(dAtA[iNdEx:postIndex])
			iNdEx = postIndex
		case 7:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field DetectionId", wireType)
			}
			m.DetectionId = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.DetectionId |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 8:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field AssociatedDetections", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
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
				return ErrInvalidLengthDetection
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthDetection
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.AssociatedDetections = append(m.AssociatedDetections, &Detection_AssociatedDetection{})
			if err := m.AssociatedDetections[len(m.AssociatedDetections)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		case 9:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field DisplayName", wireType)
			}
			var stringLen uint64
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
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
				return ErrInvalidLengthDetection
			}
			postIndex := iNdEx + intStringLen
			if postIndex < 0 {
				return ErrInvalidLengthDetection
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.DisplayName = append(m.DisplayName, string(dAtA[iNdEx:postIndex]))
			iNdEx = postIndex
		case 10:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field TimestampUsec", wireType)
			}
			m.TimestampUsec = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.TimestampUsec |= int64(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		default:
			iNdEx = preIndex
			skippy, err := skipDetection(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthDetection
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
func (m *Detection_AssociatedDetection) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowDetection
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
			return fmt.Errorf("proto: AssociatedDetection: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: AssociatedDetection: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 0 {
				return fmt.Errorf("proto: wrong wireType = %d for field Id", wireType)
			}
			m.Id = 0
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
				}
				if iNdEx >= l {
					return io.ErrUnexpectedEOF
				}
				b := dAtA[iNdEx]
				iNdEx++
				m.Id |= int32(b&0x7F) << shift
				if b < 0x80 {
					break
				}
			}
		case 2:
			if wireType != 5 {
				return fmt.Errorf("proto: wrong wireType = %d for field Confidence", wireType)
			}
			var v uint32
			if (iNdEx + 4) > l {
				return io.ErrUnexpectedEOF
			}
			v = uint32(encoding_binary.LittleEndian.Uint32(dAtA[iNdEx:]))
			iNdEx += 4
			m.Confidence = float32(math.Float32frombits(v))
		default:
			iNdEx = preIndex
			skippy, err := skipDetection(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthDetection
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
func (m *DetectionList) Unmarshal(dAtA []byte) error {
	l := len(dAtA)
	iNdEx := 0
	for iNdEx < l {
		preIndex := iNdEx
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return ErrIntOverflowDetection
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
			return fmt.Errorf("proto: DetectionList: wiretype end group for non-group")
		}
		if fieldNum <= 0 {
			return fmt.Errorf("proto: DetectionList: illegal tag %d (wire type %d)", fieldNum, wire)
		}
		switch fieldNum {
		case 1:
			if wireType != 2 {
				return fmt.Errorf("proto: wrong wireType = %d for field Detection", wireType)
			}
			var msglen int
			for shift := uint(0); ; shift += 7 {
				if shift >= 64 {
					return ErrIntOverflowDetection
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
				return ErrInvalidLengthDetection
			}
			postIndex := iNdEx + msglen
			if postIndex < 0 {
				return ErrInvalidLengthDetection
			}
			if postIndex > l {
				return io.ErrUnexpectedEOF
			}
			m.Detection = append(m.Detection, &Detection{})
			if err := m.Detection[len(m.Detection)-1].Unmarshal(dAtA[iNdEx:postIndex]); err != nil {
				return err
			}
			iNdEx = postIndex
		default:
			iNdEx = preIndex
			skippy, err := skipDetection(dAtA[iNdEx:])
			if err != nil {
				return err
			}
			if (skippy < 0) || (iNdEx+skippy) < 0 {
				return ErrInvalidLengthDetection
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
func skipDetection(dAtA []byte) (n int, err error) {
	l := len(dAtA)
	iNdEx := 0
	depth := 0
	for iNdEx < l {
		var wire uint64
		for shift := uint(0); ; shift += 7 {
			if shift >= 64 {
				return 0, ErrIntOverflowDetection
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
					return 0, ErrIntOverflowDetection
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
					return 0, ErrIntOverflowDetection
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
				return 0, ErrInvalidLengthDetection
			}
			iNdEx += length
		case 3:
			depth++
		case 4:
			if depth == 0 {
				return 0, ErrUnexpectedEndOfGroupDetection
			}
			depth--
		case 5:
			iNdEx += 4
		default:
			return 0, fmt.Errorf("proto: illegal wireType %d", wireType)
		}
		if iNdEx < 0 {
			return 0, ErrInvalidLengthDetection
		}
		if depth == 0 {
			return iNdEx, nil
		}
	}
	return 0, io.ErrUnexpectedEOF
}

var (
	ErrInvalidLengthDetection        = fmt.Errorf("proto: negative length found during unmarshaling")
	ErrIntOverflowDetection          = fmt.Errorf("proto: integer overflow")
	ErrUnexpectedEndOfGroupDetection = fmt.Errorf("proto: unexpected end of group")
)