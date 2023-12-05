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
