/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
 */

package pipelineparser_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/shared/pipelineparser"
	"gimletlabs.ai/gimlet/src/shared/uuidutils"
)

const examplePipeline = `
---
params:
- name: model
  defaultValue:
    model:
      name: yolox
- name: output_resolution
  defaultValue: "640x480"
nodes:
- name: camera_source
  kind: CameraSource
  outputs:
  - frame
- name: detect
  kind: Detect
  inputs:
    model: .params.model
    frame: .camera_source.frame
  outputs:
  - detections
- name: reclassify_detections
  kind: ForEachROI
  attributes:
    roiFunc:
      lambdaFunc:
        funcInputs:
        - roi
        funcOutputs:
        - .classify.classification
        nodes:
        - name: classify
          kind: Classify
          inputs:
            roi: .funcInputs.roi
          outputs:
          - classification
  inputs:
    frame: .camera_source.frame
    detections: .detect.detections
  outputs:
  - detections_reclassified
- name: frame_metrics_sink
  kind: FrameMetricsSink
  inputs:
    frame: .camera_source.frame
  outputs:
  - frame_metrics
- name: detection_metrics_sink
  kind: DetectionsMetricsSink
  inputs:
    detections: .reclassify_detections.detections_reclassified
- name: detections_latency_metrics_sink
  kind: LatencyMetricsSink
  attributes:
    name: detections
  inputs:
    frame: .camera_source.frame
    detections: .reclassify_detections.detections_reclassified
- name: pipeline_latency_metrics_sink
  kind: LatencyMetricsSink
  attributes:
    name: pipeline
  inputs:
    frame: .camera_source.frame
    detections: .detect.detections
    frame_metrics: .frame_metrics_sink.frame_metrics
- name: video_stream_sink
  kind: VideoStreamSink
  attributes:
    resolution: .params.output_resolution
  inputs:
    frame: .camera_source.frame
    detections: .detect.detections
    frame_metrics: .frame_metrics_sink.frame_metrics
`
const yoloxModelID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

type fakeModelResolver struct{}

func (*fakeModelResolver) GetModelID(string, string) (*typespb.UUID, error) {
	return uuidutils.ProtoFromUUIDStrOrNil(yoloxModelID), nil
}

const expectedJSON = `{
	"nodes": [
		{
			"name": "camera_source",
			"kind": "CameraSource"
		},
		{
			"name": "detect",
			"kind": "Detect",
			"metadata": {
				"model": ".params.model"
			}
		},
		{
			"name": "detection_metrics_sink",
			"kind": "DetectionsMetricsSink"
		},
		{
			"name": "detections_latency_metrics_sink",
			"kind": "LatencyMetricsSink",
			"metadata": {
				"name": "detections"
			}
		},
		{
			"name": "frame_metrics_sink",
			"kind": "FrameMetricsSink"
		},
		{
			"name": "pipeline_latency_metrics_sink",
			"kind": "LatencyMetricsSink",
			"metadata": {
				"name": "pipeline"
			}
		},
		{
			"name": "reclassify_detections",
			"kind": "ForEachROI",
			"hasChildren": true
		},
		{
			"name": "video_stream_sink",
			"kind": "VideoStreamSink"
		},
		{
			"name": "classify",
			"kind": "Classify",
			"parentName": "reclassify_detections"
		}
	],
	"edges": [
		[
			"camera_source",
			"detect"
		],
		[
			"camera_source",
			"detections_latency_metrics_sink"
		],
		[
			"camera_source",
			"frame_metrics_sink"
		],
		[
			"camera_source",
			"pipeline_latency_metrics_sink"
		],
		[
			"camera_source",
			"reclassify_detections"
		],
		[
			"camera_source",
			"video_stream_sink"
		],
		[
			"detect",
			"pipeline_latency_metrics_sink"
		],
		[
			"detect",
			"reclassify_detections"
		],
		[
			"detect",
			"video_stream_sink"
		],
		[
			"frame_metrics_sink",
			"pipeline_latency_metrics_sink"
		],
		[
			"frame_metrics_sink",
			"video_stream_sink"
		],
		[
			"reclassify_detections",
			"detection_metrics_sink"
		],
		[
			"reclassify_detections",
			"detections_latency_metrics_sink"
		]
	]
}`

func TestParser_WorksWithExample(t *testing.T) {
	parsed, err := pipelineparser.NewPipelineParser(&fakeModelResolver{}).ParsePipeline(examplePipeline)
	require.NoError(t, err)
	require.NotNil(t, parsed)
}

func TestEncoder(t *testing.T) {
	t.Run("works with example pipeline", func(t *testing.T) {
		parsed, err := pipelineparser.NewPipelineParser(&fakeModelResolver{}).ParsePipeline(examplePipeline)
		require.NoError(t, err)
		require.NotNil(t, parsed)

		encodedDAG, err := pipelineparser.GenerateJSONEncodedDAG(parsed)
		require.NoError(t, err)
		assert.Equal(t, expectedJSON, encodedDAG)
	})

	t.Run("emits empty nodes and edges if there are none", func(t *testing.T) {
		parsed := &corepb.LogicalPipeline{}
		encodedDAG, err := pipelineparser.GenerateJSONEncodedDAG(parsed)
		require.NoError(t, err)
		assert.Equal(t, `{
	"nodes": [],
	"edges": []
}`, encodedDAG)
	})
}

func TestParser_NameCollisions(t *testing.T) {
	t.Run("detects duplicate node names", func(t *testing.T) {
		_, err := pipelineparser.NewPipelineParser(&fakeModelResolver{}).ParsePipeline(`---
nodes:
- name: camera
  kind: CameraSource
- name: camera
  kind: Detect`)
		require.ErrorIs(t, err, pipelineparser.ErrDuplicateNode)
	})

	t.Run("detects duplicate in nested roi", func(t *testing.T) {
		_, err := pipelineparser.NewPipelineParser(&fakeModelResolver{}).ParsePipeline(`---
nodes:
- name: camera
  kind: CameraSource
- name: roi
  kind: ForEachROI
  attributes:
    roiFunc:
      lambdaFunc:
        nodes:
        - name: camera
          kind: Classify`)
		require.ErrorIs(t, err, pipelineparser.ErrDuplicateNode)
	})

	t.Run("doesn't error with unique names", func(t *testing.T) {
		_, err := pipelineparser.NewPipelineParser(&fakeModelResolver{}).ParsePipeline(`---
nodes:
- name: camera
  kind: CameraSource
- name: detect
  kind: Detect`)
		require.NoError(t, err)
	})
}

func TestParser_ParseParams_And_ValueTypes(t *testing.T) {
	tests := []struct {
		name              string
		yamlStr           string
		expectedParamName string
		expectedParam     *corepb.Value
	}{
		{
			name: "string param",
			yamlStr: `---
params:
- name: output_resolution
  defaultValue: "640x480"`,
			expectedParamName: "output_resolution",
			expectedParam: &corepb.Value{
				Data: &corepb.Value_StringData{
					StringData: "640x480",
				},
			},
		},
		{
			name: "int64 param",
			yamlStr: `---
params:
- name: target_fps
  defaultValue: 24`,
			expectedParamName: "target_fps",
			expectedParam: &corepb.Value{
				Data: &corepb.Value_Int64Data{
					Int64Data: 24,
				},
			},
		},
		{
			name: "bool param",
			yamlStr: `---
params:
- name: collect_metrics
  defaultValue: true`,
			expectedParamName: "collect_metrics",
			expectedParam: &corepb.Value{
				Data: &corepb.Value_BoolData{
					BoolData: true,
				},
			},
		},
		{
			name: "double param",
			yamlStr: `---
params:
- name: timestamp
  defaultValue: -12.032`,
			expectedParamName: "timestamp",
			expectedParam: &corepb.Value{
				Data: &corepb.Value_DoubleData{
					DoubleData: -12.032,
				},
			},
		},
		{
			name: "model param",
			yamlStr: `---
params:
- name: model
  defaultValue:
    model:
      name: yolox`,
			expectedParamName: "model",
			expectedParam: &corepb.Value{
				Data: &corepb.Value_ModelData{
					ModelData: &corepb.Value_ModelRef{
						Name: "yolox",
						ID:   uuidutils.ProtoFromUUIDStrOrNil(yoloxModelID),
					},
				},
			},
		},
		{
			name: "lambda param",
			yamlStr: `---
params:
- name: classifier
  defaultValue:
    lambdaFunc:
      funcInputs:
      - roi
      funcOutputs:
      - .classify.classification
      nodes:
      - name: classify
        kind: Classify
        inputs:
          classify_roi: .funcInputs.roi
        outputs:
        - classification`,
			expectedParamName: "classifier",
			expectedParam: &corepb.Value{
				Data: &corepb.Value_LambdaData{
					LambdaData: &corepb.Value_Lambda{
						Inputs:  []string{"roi"},
						Outputs: []string{".classify.classification"},
						Nodes: []*corepb.Node{
							{
								Name:       "classify",
								Kind:       corepb.LOGICAL_PIPELINE_NODE_KIND_CLASSIFY,
								Attributes: []*corepb.NodeAttributes{},
								Inputs: []*corepb.NodeInput{
									{
										Name: "classify_roi",
										Value: &corepb.NodeInput_LambdaInputValue{
											LambdaInputValue: &corepb.NodeInput_LambdaInputRef{
												Name: "roi",
											},
										},
									},
								},
								Outputs: []*corepb.NodeOutput{
									{
										Name: "classification",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			parsed, err := pipelineparser.NewPipelineParser(&fakeModelResolver{}).ParsePipeline(tc.yamlStr)
			require.NoError(t, err)
			require.NotNil(t, parsed)

			assert.Len(t, parsed.Params, 1)
			assert.Equal(t, tc.expectedParamName, parsed.Params[0].Name)
			assert.Equal(t, tc.expectedParam, parsed.Params[0].DefaultValue)
		})
	}
}

func TestParser_ParseNodes_And_InputOutput(t *testing.T) {
	tests := []struct {
		name         string
		yamlStr      string
		expectedNode *corepb.Node
		expectedErr  error
	}{
		{
			name: "camera source",
			yamlStr: `---
nodes:
- name: camera_source
  kind: CameraSource
  outputs:
  - frame`,
			expectedNode: &corepb.Node{
				Name:       "camera_source",
				Kind:       corepb.LOGICAL_PIPELINE_NODE_KIND_CAMERA_SOURCE,
				Attributes: []*corepb.NodeAttributes{},
				Inputs:     []*corepb.NodeInput{},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "frame",
					},
				},
			},
		},
		{
			name: "detect",
			yamlStr: `---
nodes:
- name: detect
  kind: Detect
  inputs:
    model: .params.my_model`,
			expectedNode: &corepb.Node{
				Name:       "detect",
				Kind:       corepb.LOGICAL_PIPELINE_NODE_KIND_DETECT,
				Attributes: []*corepb.NodeAttributes{},
				Inputs: []*corepb.NodeInput{
					{
						Name: "model",
						Value: &corepb.NodeInput_ParamValue{
							ParamValue: &corepb.ParamRef{
								Name: "my_model",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{},
			},
		},
		{
			name: "video stream sink",
			yamlStr: `---
nodes:
- name: encode
  kind: VideoStreamSink
  attributes:
    frameRate: 30
  inputs:
    frame: .camera.frame
  outputs:
  - encoded_stream
  - frame_metrics`,
			expectedNode: &corepb.Node{
				Name: "encode",
				Kind: corepb.LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK,
				Attributes: []*corepb.NodeAttributes{
					{
						Name: "frameRate",
						Value: &corepb.Value{
							Data: &corepb.Value_Int64Data{
								Int64Data: 30,
							},
						},
					},
				},
				Inputs: []*corepb.NodeInput{
					{
						Name: "frame",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "camera",
								Name:     "frame",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "encoded_stream",
					},
					{
						Name: "frame_metrics",
					},
				},
			},
		},
		{
			name: "frame metrics sink",
			yamlStr: `---
nodes:
- name: frame_sink
  kind: FrameMetricsSink
  inputs:
    frame: .camera.frame`,
			expectedNode: &corepb.Node{
				Name:       "frame_sink",
				Kind:       corepb.LOGICAL_PIPELINE_NODE_KIND_FRAME_METRICS_SINK,
				Attributes: []*corepb.NodeAttributes{},
				Inputs: []*corepb.NodeInput{
					{
						Name: "frame",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "camera",
								Name:     "frame",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{},
			},
		},
		{
			name: "video stream sink",
			yamlStr: `---
nodes:
- name: video_stream_sink
  kind: VideoStreamSink
  attributes:
    resolution: .params.output_resolution`,
			expectedNode: &corepb.Node{
				Name: "video_stream_sink",
				Kind: corepb.LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK,
				Attributes: []*corepb.NodeAttributes{
					{
						Name: "resolution",
						Value: &corepb.Value{
							Data: &corepb.Value_ParamData{
								ParamData: &corepb.ParamRef{
									Name: "output_resolution",
								},
							},
						},
					},
				},
				Inputs:  []*corepb.NodeInput{},
				Outputs: []*corepb.NodeOutput{},
			},
		},
		{
			name: "bad node",
			yamlStr: `---
nodes:
- name: bad_node
  kind: Detect
  inputs:
    frame: .funcInputs.frame`,
			expectedErr: pipelineparser.ErrUnsupportedValue,
		},
		{
			name: "duplicate node",
			yamlStr: `---
nodes:
- name: duplicate_node
  kind: CameraSource
- name: duplicate_node
  kind: Detect`,
			expectedErr: pipelineparser.ErrDuplicateNode,
		},
		{
			name: "segment node",
			yamlStr: `---
nodes:
- name: segment
  kind: Segment
  inputs:
    model: .params.my_model`,
			expectedNode: &corepb.Node{
				Name:       "segment",
				Kind:       corepb.LOGICAL_PIPELINE_NODE_KIND_SEGMENT,
				Attributes: []*corepb.NodeAttributes{},
				Inputs: []*corepb.NodeInput{
					{
						Name: "model",
						Value: &corepb.NodeInput_ParamValue{
							ParamValue: &corepb.ParamRef{
								Name: "my_model",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{},
			},
		},
		{
			name: "regress node",
			yamlStr: `---
nodes:
- name: regress
  kind: Regress
  attributes:
    model: .params.my_model
  inputs:
    frame: .camera.frame
  outputs:
  - regression`,
			expectedNode: &corepb.Node{
				Name: "regress",
				Kind: corepb.LOGICAL_PIPELINE_NODE_KIND_REGRESS,
				Attributes: []*corepb.NodeAttributes{
					{
						Name: "model",
						Value: &corepb.Value{
							Data: &corepb.Value_ParamData{
								ParamData: &corepb.ParamRef{
									Name: "my_model",
								},
							},
						},
					},
				},
				Inputs: []*corepb.NodeInput{
					{
						Name: "frame",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "camera",
								Name:     "frame",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "regression",
					},
				},
			},
		},
		{
			name: "multi purpose model node",
			yamlStr: `---
nodes:
- name: multi_purpose_model
  kind: MultiPurposeModel
  attributes:
    classification_tensor_idx: 0
    regression_tensor_idx: 1
    model: .params.my_model
  inputs:
    frame: .camera.frame
  outputs:
  - classification
  - regression`,
			expectedNode: &corepb.Node{
				Name: "multi_purpose_model",
				Kind: corepb.LOGICAL_PIPELINE_NODE_KIND_MULTI_PURPOSE_MODEL,
				Attributes: []*corepb.NodeAttributes{
					{
						Name: "classification_tensor_idx",
						Value: &corepb.Value{
							Data: &corepb.Value_Int64Data{
								Int64Data: 0,
							},
						},
					},
					{
						Name: "model",
						Value: &corepb.Value{
							Data: &corepb.Value_ParamData{
								ParamData: &corepb.ParamRef{
									Name: "my_model",
								},
							},
						},
					},
					{
						Name: "regression_tensor_idx",
						Value: &corepb.Value{
							Data: &corepb.Value_Int64Data{
								Int64Data: 1,
							},
						},
					},
				},
				Inputs: []*corepb.NodeInput{
					{
						Name: "frame",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "camera",
								Name:     "frame",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "classification",
					},
					{
						Name: "regression",
					},
				},
			},
		},
		{
			name: "generate text",
			yamlStr: `---
nodes:
- name: generate
  kind: GenerateTokens
  attributes:
    model: .params.my_model
  inputs:
    tokens: .tokenizer.tokens
  outputs:
  - generated_tokens`,
			expectedNode: &corepb.Node{
				Name: "generate",
				Kind: corepb.LOGICAL_PIPELINE_NODE_KIND_GENERATE_TOKENS,
				Attributes: []*corepb.NodeAttributes{
					{
						Name: "model",
						Value: &corepb.Value{
							Data: &corepb.Value_ParamData{
								ParamData: &corepb.ParamRef{
									Name: "my_model",
								},
							},
						},
					},
				},
				Inputs: []*corepb.NodeInput{
					{
						Name: "tokens",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "tokenizer",
								Name:     "tokens",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "generated_tokens",
					},
				},
			},
		},
		{
			name: "tokenize",
			yamlStr: `---
nodes:
- name: tokenize
  kind: Tokenize
  attributes:
    tokenizer: .params.my_tokenizer
  inputs:
    text: .text_source.text
  outputs:
  - tokens`,
			expectedNode: &corepb.Node{
				Name: "tokenize",
				Kind: corepb.LOGICAL_PIPELINE_NODE_KIND_TOKENIZE,
				Attributes: []*corepb.NodeAttributes{
					{
						Name: "tokenizer",
						Value: &corepb.Value{
							Data: &corepb.Value_ParamData{
								ParamData: &corepb.ParamRef{
									Name: "my_tokenizer",
								},
							},
						},
					},
				},
				Inputs: []*corepb.NodeInput{
					{
						Name: "text",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "text_source",
								Name:     "text",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "tokens",
					},
				},
			},
		},
		{
			name: "detokenize",
			yamlStr: `---
nodes:
- name: detokenize
  kind: Detokenize
  attributes:
    tokenizer: .params.my_tokenizer
  inputs:
    tokens: .model.tokens
  outputs:
  - text`,
			expectedNode: &corepb.Node{
				Name: "detokenize",
				Kind: corepb.LOGICAL_PIPELINE_NODE_KIND_DETOKENIZE,
				Attributes: []*corepb.NodeAttributes{
					{
						Name: "tokenizer",
						Value: &corepb.Value{
							Data: &corepb.Value_ParamData{
								ParamData: &corepb.ParamRef{
									Name: "my_tokenizer",
								},
							},
						},
					},
				},
				Inputs: []*corepb.NodeInput{
					{
						Name: "tokens",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "model",
								Name:     "tokens",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "text",
					},
				},
			},
		},
		{
			name: "track",
			yamlStr: `---
nodes:
- name: track
  kind: Track
  inputs:
    detections: .detect.detections
  outputs:
  - tracked_detections`,
			expectedNode: &corepb.Node{
				Name:       "track",
				Kind:       corepb.LOGICAL_PIPELINE_NODE_KIND_TRACK,
				Attributes: []*corepb.NodeAttributes{},
				Inputs: []*corepb.NodeInput{
					{
						Name: "detections",
						Value: &corepb.NodeInput_NodeOutputValue{
							NodeOutputValue: &corepb.NodeInput_NodeOutputRef{
								NodeName: "detect",
								Name:     "detections",
							},
						},
					},
				},
				Outputs: []*corepb.NodeOutput{
					{
						Name: "tracked_detections",
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			parsed, err := pipelineparser.NewPipelineParser(&fakeModelResolver{}).ParsePipeline(tc.yamlStr)
			if tc.expectedErr != nil {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.expectedErr)
			} else {
				require.NoError(t, err)
			}
			if tc.expectedNode != nil {
				require.NotNil(t, parsed)

				assert.Len(t, parsed.Nodes, 1)
				assert.Equal(t, tc.expectedNode, parsed.Nodes[0])
			}
		})
	}
}
