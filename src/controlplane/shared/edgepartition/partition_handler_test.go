/*
 * Copyright 2023- Gimlet Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package edgepartition_test

import (
	"sync"
	"testing"

	"github.com/gofrs/uuid/v5"
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/types"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
	"gimletlabs.ai/gimlet/src/shared/services/natstest"
	utils "gimletlabs.ai/gimlet/src/shared/uuidutils"
)

type handler struct {
	handleMsg func(*corepb.EdgeCPMetadata, *types.Any, string)
}

func (h *handler) HandleMessage(md *corepb.EdgeCPMetadata, msg *types.Any, msgType string) {
	h.handleMsg(md, msg, msgType)
}

func TestPartitionHandler_MessageHandler(t *testing.T) {
	viper.Set("partition_id", 0)
	viper.Set("partition_count", 1)

	nc := natstest.MustStartTestNATS(t)

	deviceID := uuid.Must(uuid.NewV4())

	var wg sync.WaitGroup

	wg.Add(1)
	h := &handler{
		handleMsg: func(md *corepb.EdgeCPMetadata, msg *types.Any, msgType string) {
			assert.Equal(t, corepb.EDGE_CP_TOPIC_STATUS, md.Topic)
			assert.Equal(t, utils.ProtoFromUUID(deviceID), md.DeviceID)
			assert.Equal(t, proto.MessageName(&corepb.EdgeHeartbeat{}), msgType)
			hbMsg := &corepb.EdgeHeartbeat{}
			err := types.UnmarshalAny(msg, hbMsg)
			require.NoError(t, err)
			assert.Equal(t, int64(10), hbMsg.SeqID)
			wg.Done()
		},
	}

	s := edgepartition.NewPartitionHandler(nc)
	err := s.RegisterHandler(corepb.EDGE_CP_TOPIC_STATUS, func() edgepartition.MessageHandler {
		return h
	}, &edgepartition.MessageHandlerConfig{
		NumGoroutines: 1,
	})
	require.NoError(t, err)

	anyMsg, err := types.MarshalAny(&corepb.EdgeHeartbeat{SeqID: 10})
	require.NoError(t, err)
	msg := &corepb.EdgeCPMessage{
		Metadata: &corepb.EdgeCPMetadata{
			Topic:         corepb.EDGE_CP_TOPIC_STATUS,
			DeviceID:      utils.ProtoFromUUID(deviceID),
			RecvTimestamp: nil,
		},
		Msg: anyMsg,
	}
	b, err := msg.Marshal()
	require.NoError(t, err)

	topic, err := edgepartition.EdgeToCPNATSTopic(deviceID, corepb.EDGE_CP_TOPIC_STATUS, false)
	require.NoError(t, err)
	err = nc.Publish(topic, b)
	require.NoError(t, err)

	wg.Wait()
}
