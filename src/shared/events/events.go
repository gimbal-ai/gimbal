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

package events

// This file tracks all the events the backend produces to
// reduce the chance of a typo messing up the analytics.

const (
	UserLoggedIn            = "ev-user-logged-in"
	UserSignedUp            = "ev-user-signed-up"
	OrgCreated              = "ev-org-created"
	FleetCreated            = "ev-fleet-created"
	ModelCreated            = "ev-model-created"
	LogicalPipelineCreated  = "ev-logical-pipeline-created"
	LogicalPipelineDeployed = "ev-logical-pipeline-deployed"
)
