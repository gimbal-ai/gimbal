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

package resolve

// Node is the interface needed for the nodes in a graph in order to use DFS to search that graph.
// The Children method allows graphs to be computed lazily as DFS requests children,
// rather than needing to specify the complete graph ahead of time.
type Node[T any] interface {
	Name() string
	Children() ([]T, error)
}

// DFS performs depth first search on a graph of nodes, defined by the Node interface.
type DFS[T Node[T]] struct {
	stack   []T
	visited map[string]bool
}

// NewDFS creates a new DFS searcher object.
func NewDFS[T Node[T]]() *DFS[T] {
	return &DFS[T]{
		stack:   make([]T, 0),
		visited: make(map[string]bool),
	}
}

func (d *DFS[T]) Push(node T) {
	d.stack = append(d.stack, node)
	d.visited[node.Name()] = true
}

func (d *DFS[T]) empty() bool {
	return len(d.stack) == 0
}

func (d *DFS[T]) pop() {
	if d.empty() {
		return
	}
	d.stack = d.stack[:len(d.stack)-1]
}

func (d *DFS[T]) peek() T {
	return d.stack[len(d.stack)-1]
}

// PostOrderTraverse does a depth-first traversal of the graph calling `postCB`
// after each node in the graph has been visited and all of its descendents have been visited.
// For nodes in a cycle, the order of `postCB` calls is arbitrary within the cycle.
func (d *DFS[T]) PostOrderTraverse(postCB func(T) error) error {
	for !d.empty() {
		n := d.peek()
		tail := true
		cs, err := n.Children()
		if err != nil {
			return err
		}
		for _, child := range cs {
			if d.visited[child.Name()] {
				continue
			}
			tail = false
			d.Push(child)
		}
		if tail {
			d.pop()
			if err := postCB(n); err != nil {
				return err
			}
		}
	}
	return nil
}
