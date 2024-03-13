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

package fmpb

import (
	"database/sql/driver"
	"fmt"
)

func (p *OSKind) Scan(src any) error {
	s, ok := src.(string)
	if !ok {
		return fmt.Errorf("unexpected type for OSKind: %T", src)
	}
	i, ok := OSKind_value[s]
	if !ok {
		return fmt.Errorf("invalid OSKind value: %s", s)
	}
	*p = OSKind(i)
	return nil
}

func (p OSKind) Value() (driver.Value, error) {
	return p.String(), nil
}
