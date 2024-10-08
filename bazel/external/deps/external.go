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

package deps

import (
	// Indirectly used by github.com/stackb/rules_proto/pkg/protoc.
	_ "github.com/bazelbuild/buildtools/build"
	// Used for gazelle plugin.
	_ "github.com/bmatcuk/doublestar"
	// We use the yq cli for bazel commands.
	_ "github.com/mikefarah/yq/v4/pkg/yqlib"
	// Used by our proto libs.
	_ "google.golang.org/protobuf/types/known/anypb"
)
