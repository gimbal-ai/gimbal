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

package pipelineparser

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/common/typespb"
)

var (
	ErrUnsupportedValue = errors.New("unsupported value")
	ErrDuplicateNode    = errors.New("duplicate node names detected")
)

// ModelResolver interfaces provides PipelineParser a way to resolve model (org, name) pairs into model IDs.
type ModelResolver interface {
	GetModelID(name string, orgName string) (*typespb.UUID, error)
}

// PipelineParser parses pipeline yamls into LogicalPipeline protobufs.
type PipelineParser struct {
	mr       ModelResolver
	modelIDs []*typespb.UUID
}

// NewPipelineParser creates a new PipelineParser with the given ModelResolver.
func NewPipelineParser(mr ModelResolver) *PipelineParser {
	return &PipelineParser{
		mr: mr,
	}
}

type param struct {
	Name         string `yaml:"name"`
	DefaultValue any    `yaml:"defaultValue"`
}

type node struct {
	Name       string            `yaml:"name"`
	Kind       string            `yaml:"kind"`
	Attributes map[string]any    `yaml:"attributes"`
	Inputs     map[string]string `yaml:"inputs"`
	Outputs    []string          `yaml:"outputs"`
}

type lambda struct {
	FuncInputs  []string `yaml:"funcInputs"`
	FuncOutputs []string `yaml:"funcOutputs"`
	Nodes       []node   `yaml:"nodes"`
}

type pipeline struct {
	Params []param `yaml:"params"`
	Nodes  []node  `yaml:"nodes"`
}

type modelRef struct {
	Name string `yaml:"name"`
	Org  string `yaml:"org"`
}

func (p *PipelineParser) convertModelRef(v any) (*corepb.Value, error) {
	modelYaml, err := yaml.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("%w: couldn't convert model %v", ErrUnsupportedValue, v)
	}
	m := &modelRef{}
	err = yaml.Unmarshal(modelYaml, m)
	if err != nil {
		return nil, fmt.Errorf("%w: couldn't convert model %v", ErrUnsupportedValue, v)
	}

	modelID, err := p.mr.GetModelID(m.Name, m.Org)
	if err != nil {
		return nil, fmt.Errorf("%w: couldn't convert model %v", ErrUnsupportedValue, v)
	}
	p.modelIDs = append(p.modelIDs, modelID)
	return &corepb.Value{Data: &corepb.Value_ModelData{ModelData: &corepb.Value_ModelRef{Name: m.Name, ID: modelID}}}, nil
}

func (p *PipelineParser) convertLambdaFunc(v any) (*corepb.Value, error) {
	lambdaYaml, err := yaml.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("%w: couldn't convert lambdaFunc %v", ErrUnsupportedValue, v)
	}

	lf := &lambda{}
	err = yaml.Unmarshal(lambdaYaml, lf)
	if err != nil {
		return nil, fmt.Errorf("%w: couldn't unmarshal lambdaFunc node %v", ErrUnsupportedValue, v)
	}

	nodes, err := p.convertNodes(lf.Nodes)
	if err != nil {
		return nil, err
	}

	return &corepb.Value{Data: &corepb.Value_LambdaData{LambdaData: &corepb.Value_Lambda{
		Inputs:  lf.FuncInputs,
		Outputs: lf.FuncOutputs,
		Nodes:   nodes,
	}}}, nil
}

func (p *PipelineParser) convertValue(in any) (*corepb.Value, error) {
	switch v := in.(type) {
	case int64:
		return &corepb.Value{Data: &corepb.Value_Int64Data{Int64Data: v}}, nil
	case int32:
		return &corepb.Value{Data: &corepb.Value_Int64Data{Int64Data: int64(v)}}, nil
	case int16:
		return &corepb.Value{Data: &corepb.Value_Int64Data{Int64Data: int64(v)}}, nil
	case int8:
		return &corepb.Value{Data: &corepb.Value_Int64Data{Int64Data: int64(v)}}, nil
	case int:
		return &corepb.Value{Data: &corepb.Value_Int64Data{Int64Data: int64(v)}}, nil
	case float64:
		return &corepb.Value{Data: &corepb.Value_DoubleData{DoubleData: v}}, nil
	case bool:
		return &corepb.Value{Data: &corepb.Value_BoolData{BoolData: v}}, nil
	case string:
		if strings.HasPrefix(v, ".param") {
			paramName := strings.TrimPrefix(v, ".params.")
			return &corepb.Value{Data: &corepb.Value_ParamData{ParamData: &corepb.ParamRef{Name: paramName}}}, nil
		}
		return &corepb.Value{Data: &corepb.Value_StringData{StringData: v}}, nil
	case map[string]any:
		for key, val := range v {
			if key == "model" {
				return p.convertModelRef(val)
			}
			if key == "lambdaFunc" {
				return p.convertLambdaFunc(val)
			}
			return nil, fmt.Errorf("%w: unsupported key in map value %s", ErrUnsupportedValue, key)
		}
		return nil, fmt.Errorf("%w: empty map in value", ErrUnsupportedValue)
	default:
		return nil, fmt.Errorf("%w: unsupported type %v", ErrUnsupportedValue, v)
	}
}

func convertNodeKind(kind string) (corepb.LogicalPipelineNodeKind, error) {
	switch kind {
	case "CameraSource":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_CAMERA_SOURCE, nil
	case "Detect":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_DETECT, nil
	case "Classify":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_CLASSIFY, nil
	case "Segment":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_SEGMENT, nil
	case "Track":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_TRACK, nil
	case "Regress":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_REGRESS, nil
	case "MultiPurposeModel":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_MULTI_PURPOSE_MODEL, nil
	case "ForEachROI":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_FOR_EACH_ROI, nil
	case "VideoStreamSink":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK, nil
	case "DetectionsMetricsSink":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_DETECTIONS_METRICS_SINK, nil
	case "LatencyMetricsSink":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_LATENCY_METRICS_SINK, nil
	case "FrameMetricsSink":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_FRAME_METRICS_SINK, nil
	case "TextStreamSink":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SINK, nil
	case "TextStreamSource":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SOURCE, nil
	case "GenerateTokens":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_GENERATE_TOKENS, nil
	case "Tokenize":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_TOKENIZE, nil
	case "Detokenize":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_DETOKENIZE, nil
	case "Embed":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_EMBED, nil
	case "VectorSearch":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_VECTOR_SEARCH, nil
	case "TemplateChatMessage":
		return corepb.LOGICAL_PIPELINE_NODE_KIND_TEMPLATE_CHAT_MESSAGE, nil
	default:
		return corepb.LOGICAL_PIPELINE_NODE_KIND_UNKNOWN, fmt.Errorf("%w: unsupported node kind %s", ErrUnsupportedValue, kind)
	}
}

func (p *PipelineParser) convertNodeAttributes(attrs map[string]any) ([]*corepb.NodeAttributes, error) {
	converted := make([]*corepb.NodeAttributes, 0, len(attrs))
	for key, val := range attrs {
		convertedVal, err := p.convertValue(val)
		if err != nil {
			return nil, err
		}
		converted = append(converted, &corepb.NodeAttributes{
			Name:  key,
			Value: convertedVal,
		})
	}

	sort.Slice(converted, func(i, j int) bool {
		return converted[i].Name < converted[j].Name
	})
	return converted, nil
}

func (*PipelineParser) convertParamRef(v string) (*corepb.NodeInput_ParamValue, error) {
	if !strings.HasPrefix(v, ".params") {
		return nil, fmt.Errorf("%w: invalid param reference %s", ErrUnsupportedValue, v)
	}
	paramName := strings.TrimPrefix(v, ".params.")
	return &corepb.NodeInput_ParamValue{ParamValue: &corepb.ParamRef{Name: paramName}}, nil
}

func (*PipelineParser) convertNodeOutputRef(v string) (*corepb.NodeInput_NodeOutputValue, error) {
	if !strings.HasPrefix(v, ".") || strings.HasPrefix(v, ".params") || strings.HasPrefix(v, ".funcInputs") {
		return nil, fmt.Errorf("%w: invalid node output reference %s", ErrUnsupportedValue, v)
	}

	parts := strings.Split(v, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("%w: invalid input value %s", ErrUnsupportedValue, v)
	}
	return &corepb.NodeInput_NodeOutputValue{NodeOutputValue: &corepb.NodeInput_NodeOutputRef{NodeName: parts[1], Name: parts[2]}}, nil
}

func (*PipelineParser) convertNodeLambdaInput(v string) (*corepb.NodeInput_LambdaInputValue, error) {
	if !strings.HasPrefix(v, ".funcInputs") {
		return nil, fmt.Errorf("%w: invalid lambda func input reference %s", ErrUnsupportedValue, v)
	}
	inputName := strings.TrimPrefix(v, ".funcInputs.")
	return &corepb.NodeInput_LambdaInputValue{LambdaInputValue: &corepb.NodeInput_LambdaInputRef{Name: inputName}}, nil
}

func (p *PipelineParser) convertNodeInputs(inputs map[string]string) ([]*corepb.NodeInput, error) {
	converted := make([]*corepb.NodeInput, 0, len(inputs))
	for key, val := range inputs {
		if strings.HasPrefix(val, ".params") {
			paramRef, err := p.convertParamRef(val)
			if err != nil {
				return nil, err
			}
			converted = append(converted, &corepb.NodeInput{
				Name:  key,
				Value: paramRef,
			})
			continue
		}

		if strings.HasPrefix(val, ".funcInputs") {
			paramRef, err := p.convertNodeLambdaInput(val)
			if err != nil {
				return nil, err
			}
			converted = append(converted, &corepb.NodeInput{
				Name:  key,
				Value: paramRef,
			})
			continue
		}

		nodeOutputRef, err := p.convertNodeOutputRef(val)
		if err != nil {
			return nil, err
		}
		converted = append(converted, &corepb.NodeInput{
			Name:  key,
			Value: nodeOutputRef,
		})
	}

	sort.Slice(converted, func(i, j int) bool {
		return converted[i].Name < converted[j].Name
	})
	return converted, nil
}

func (*PipelineParser) convertNodeOutputs(outputs []string) []*corepb.NodeOutput {
	converted := make([]*corepb.NodeOutput, len(outputs))
	for i, val := range outputs {
		converted[i] = &corepb.NodeOutput{
			Name: val,
		}
	}
	return converted
}

func (p *PipelineParser) convertNodes(nodes []node) ([]*corepb.Node, error) {
	converted := make([]*corepb.Node, len(nodes))
	for i, node := range nodes {
		attrs, err := p.convertNodeAttributes(node.Attributes)
		if err != nil {
			return nil, err
		}
		inputs, err := p.convertNodeInputs(node.Inputs)
		if err != nil {
			return nil, err
		}

		kind, err := convertNodeKind(node.Kind)
		if err != nil {
			return nil, err
		}

		converted[i] = &corepb.Node{
			Name:       node.Name,
			Kind:       kind,
			Attributes: attrs,
			Inputs:     inputs,
			Outputs:    p.convertNodeOutputs(node.Outputs),
		}
	}
	return converted, nil
}

func hasLambdaInputs(nodes []*corepb.Node) bool {
	for _, node := range nodes {
		for _, input := range node.Inputs {
			if input.GetLambdaInputValue() != nil {
				return true
			}
		}
	}
	return false
}

func getNodeNames(nodes []*corepb.Node) []string {
	var names []string
	for _, node := range nodes {
		names = append(names, node.Name)
		for _, attr := range node.Attributes {
			if attr.Value.GetLambdaData() != nil {
				names = append(names, getNodeNames(attr.Value.GetLambdaData().Nodes)...)
			}
		}
	}
	return names
}

func checkNameCollisions(nodes []*corepb.Node) error {
	seen := make(map[string]bool)
	for _, name := range getNodeNames(nodes) {
		if seen[name] {
			return fmt.Errorf("%w: node names must be unique, %s seen multiple times", ErrDuplicateNode, name)
		}
		seen[name] = true
	}
	return nil
}

// ParsePipeline parses a pipeline YAML into a LogicalPipeline protobuf.
func (p *PipelineParser) ParsePipeline(pipelineYAML string) (*corepb.LogicalPipeline, error) {
	pipe := &pipeline{}
	err := yaml.Unmarshal([]byte(pipelineYAML), pipe)
	if err != nil {
		return nil, err
	}

	lpb := &corepb.LogicalPipeline{}
	for _, param := range pipe.Params {
		defVal, err := p.convertValue(param.DefaultValue)
		if err != nil {
			return nil, err
		}
		lpb.Params = append(lpb.Params, &corepb.Param{
			Name:         param.Name,
			DefaultValue: defVal,
		})
	}
	nodes, err := p.convertNodes(pipe.Nodes)
	if err != nil {
		return nil, err
	}
	if hasLambdaInputs(nodes) {
		return nil, fmt.Errorf("%w: lambda inputs are not supported in the top level graph", ErrUnsupportedValue)
	}
	err = checkNameCollisions(nodes)
	if err != nil {
		return nil, err
	}
	lpb.Nodes = nodes

	lpb.ModelIDs = p.modelIDs
	return lpb, nil
}

type nodeKind corepb.LogicalPipelineNodeKind

type dagNode struct {
	Name        string         `json:"name"`
	Kind        nodeKind       `json:"kind"`
	ParentName  string         `json:"parentName,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	HasChildren bool           `json:"hasChildren,omitempty"`
}

type dag struct {
	Nodes []*dagNode `json:"nodes"`
	Edges [][]string `json:"edges"`
}

func (k nodeKind) MarshalJSON() ([]byte, error) {
	switch corepb.LogicalPipelineNodeKind(k) {
	case corepb.LOGICAL_PIPELINE_NODE_KIND_CAMERA_SOURCE:
		return json.Marshal("CameraSource")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_DETECT:
		return json.Marshal("Detect")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_CLASSIFY:
		return json.Marshal("Classify")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_SEGMENT:
		return json.Marshal("Segment")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_TRACK:
		return json.Marshal("Track")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_MULTI_PURPOSE_MODEL:
		return json.Marshal("MultiPurposeModel")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_REGRESS:
		return json.Marshal("Regress")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_FOR_EACH_ROI:
		return json.Marshal("ForEachROI")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK:
		return json.Marshal("VideoStreamSink")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_DETECTIONS_METRICS_SINK:
		return json.Marshal("DetectionsMetricsSink")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_LATENCY_METRICS_SINK:
		return json.Marshal("LatencyMetricsSink")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_FRAME_METRICS_SINK:
		return json.Marshal("FrameMetricsSink")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SINK:
		return json.Marshal("TextStreamSink")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SOURCE:
		return json.Marshal("TextStreamSource")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_GENERATE_TOKENS:
		return json.Marshal("GenerateTokens")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_TOKENIZE:
		return json.Marshal("Tokenize")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_EMBED:
		return json.Marshal("Embed")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_DETOKENIZE:
		return json.Marshal("Detokenize")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_VECTOR_SEARCH:
		return json.Marshal("VectorSearch")
	case corepb.LOGICAL_PIPELINE_NODE_KIND_TEMPLATE_CHAT_MESSAGE:
		return json.Marshal("TemplateChatMessage")
	default:
		return nil, fmt.Errorf("%w: unsupported node kind %d", ErrUnsupportedValue, k)
	}
}

func convertNodeForDag(node *corepb.Node) ([]*dagNode, [][]string) {
	var newNodes []*dagNode
	var newEdges [][]string

	currNode := &dagNode{
		Name:        node.Name,
		Kind:        nodeKind(node.Kind),
		Metadata:    make(map[string]any),
		HasChildren: false,
	}

	for _, in := range node.Inputs {
		if in.GetNodeOutputValue() != nil {
			newEdges = append(newEdges, []string{in.GetNodeOutputValue().NodeName, node.Name})
		}
		if in.GetParamValue() != nil {
			currNode.Metadata[in.Name] = fmt.Sprintf(".params.%s", in.GetParamValue().Name)
		}
	}

	for _, attr := range node.Attributes {
		if attr.Value.GetStringData() != "" {
			currNode.Metadata[attr.Name] = attr.Value.GetStringData()
		}
		if attr.Value.GetInt64Data() != 0 {
			currNode.Metadata[attr.Name] = attr.Value.GetInt64Data()
		}
		if attr.Value.GetDoubleData() != 0 {
			currNode.Metadata[attr.Name] = attr.Value.GetDoubleData()
		}
		if attr.Value.GetBoolData() {
			currNode.Metadata[attr.Name] = attr.Value.GetBoolData()
		}
		if attr.Value.GetModelData() != nil {
			currNode.Metadata[attr.Name] = fmt.Sprintf(".model.%s", attr.Value.GetModelData().Name)
		}
		if attr.Value.GetLambdaData() != nil {
			currNode.HasChildren = true
			for _, n := range attr.Value.GetLambdaData().Nodes {
				lambdaNodes, lambdaEdges := convertNodeForDag(n)
				for _, ln := range lambdaNodes {
					ln.ParentName = node.Name
				}
				newNodes = append(newNodes, lambdaNodes...)
				newEdges = append(newEdges, lambdaEdges...)
			}
		}
	}

	newNodes = append(newNodes, currNode)
	return newNodes, newEdges
}

func GenerateJSONEncodedDAG(pipeline *corepb.LogicalPipeline) (string, error) {
	pipelineDAG := &dag{
		Nodes: []*dagNode{},
		Edges: [][]string{},
	}
	err := checkNameCollisions(pipeline.Nodes)
	if err != nil {
		return "", err
	}
	for _, node := range pipeline.Nodes {
		currNodes, currEdges := convertNodeForDag(node)

		pipelineDAG.Nodes = append(pipelineDAG.Nodes, currNodes...)
		pipelineDAG.Edges = append(pipelineDAG.Edges, currEdges...)
	}

	// Sort the nodes and edges for deterministic output.
	sort.Slice(pipelineDAG.Nodes, func(i, j int) bool {
		if pipelineDAG.Nodes[i].ParentName == "" && pipelineDAG.Nodes[j].ParentName != "" {
			return true
		}
		if pipelineDAG.Nodes[i].ParentName != "" && pipelineDAG.Nodes[j].ParentName == "" {
			return false
		}
		return pipelineDAG.Nodes[i].Name < pipelineDAG.Nodes[j].Name
	})
	sort.Slice(pipelineDAG.Edges, func(i, j int) bool {
		if pipelineDAG.Edges[i][0] == pipelineDAG.Edges[j][0] {
			return pipelineDAG.Edges[i][1] < pipelineDAG.Edges[j][1]
		}
		return pipelineDAG.Edges[i][0] < pipelineDAG.Edges[j][0]
	})

	enc, err := json.MarshalIndent(pipelineDAG, "", "\t")
	if err != nil {
		return "", err
	}
	return string(enc), nil
}
