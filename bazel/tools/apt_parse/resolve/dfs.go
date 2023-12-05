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
