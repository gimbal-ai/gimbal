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

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gogo/protobuf/proto"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	"gimletlabs.ai/gimlet/src/common/typespb"
	// TODO(philkuz): move the parser out of controlplane.
	"gimletlabs.ai/gimlet/src/controlplane/logicalpipeline/controllers"
)

// mockModelResolver implements a simple mock for the modelResolver interface.
type mockModelResolver struct{}

func (*mockModelResolver) GetModelID(_, _ string) (*typespb.UUID, error) {
	// For simplicity, return a standalone UUID
	return &typespb.UUID{HighBits: 1, LowBits: 2}, nil
}

func init() {
	pflag.String("yaml", "", "Path to the YAML file to parse")
	pflag.String("output", "", "Path to the output file")
}

func main() {
	pflag.Parse()

	// Must call after all flags are setup.
	viper.AutomaticEnv()
	viper.SetEnvPrefix("GML")
	_ = viper.BindPFlags(pflag.CommandLine)

	yamlFile := viper.GetString("yaml")
	outputFile := viper.GetString("output")

	if yamlFile == "" {
		log.Fatal("Please provide a YAML file using the --yaml flag")
	}

	if outputFile == "" {
		log.Fatal("Please provide an output file using the --output flag")
	}

	yamlContent, err := os.ReadFile(yamlFile)
	if err != nil {
		log.Fatalf("Error reading YAML file: %v", err)
	}

	// Create a mock modelResolver for parsing
	mockResolver := &mockModelResolver{}

	parser := controllers.NewPipelineParser(mockResolver)
	parsedPipeline, err := parser.ParsePipeline(string(yamlContent))
	if err != nil {
		log.Fatalf("Error parsing pipeline YAML: %v", err)
	}

	// Convert the parsed pipeline to proto text format
	protoText := proto.MarshalTextString(parsedPipeline)

	// Write the proto text to the specified output file
	err = os.WriteFile(outputFile, []byte(protoText), 0o644)
	if err != nil {
		log.Fatalf("Error writing to output file: %v", err)
	}
	fmt.Printf("Proto text written to %s\n", outputFile)
}
