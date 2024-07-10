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

package resolve_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/resolve"
)

type nodeImpl struct {
	id       int
	children []int
	allNodes map[int]*nodeImpl
}

func (n *nodeImpl) Name() string {
	return fmt.Sprintf("%d", n.id)
}

func (n *nodeImpl) Children() ([]*nodeImpl, error) {
	children := make([]*nodeImpl, len(n.children))
	for i, id := range n.children {
		children[i] = n.allNodes[id]
	}
	return children, nil
}

func TestDFS(t *testing.T) {
	testCases := []struct {
		name               string
		allNodes           []*nodeImpl
		startNodes         []int
		expectedVisitOrder []int
	}{
		{
			name: "simple dag PostOrderTraverse",
			allNodes: []*nodeImpl{
				{
					id:       0,
					children: []int{1},
				},
				{
					id:       1,
					children: []int{2},
				},
				{
					id:       2,
					children: []int{},
				},
			},
			startNodes:         []int{0},
			expectedVisitOrder: []int{2, 1, 0},
		},
		{
			name: "cyclic graph PostOrderTraverse",
			allNodes: []*nodeImpl{
				{
					id:       0,
					children: []int{1},
				},
				{
					id:       1,
					children: []int{0},
				},
			},
			startNodes:         []int{0},
			expectedVisitOrder: []int{1, 0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			allNodes := make(map[int]*nodeImpl)
			for _, n := range tc.allNodes {
				n.allNodes = allNodes
				allNodes[n.id] = n
			}

			dfs := resolve.NewDFS[*nodeImpl]()
			for _, id := range tc.startNodes {
				dfs.Push(allNodes[id])
			}
			order := []int{}
			cb := func(n *nodeImpl) error {
				order = append(order, n.id)
				return nil
			}
			err := dfs.PostOrderTraverse(cb)
			require.NoError(t, err)

			assert.Equal(t, tc.expectedVisitOrder, order)
		})
	}
}
