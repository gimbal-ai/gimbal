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

package vectorutils

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

type EmbeddingVector []float32

func (e EmbeddingVector) Stringify() string {
	var embeddingStrings []string
	for _, eVal := range e {
		embeddingStrings = append(embeddingStrings, strconv.FormatFloat(float64(eVal), 'f', -1, 32))
	}

	return fmt.Sprintf("[%s]", strings.Join(embeddingStrings, ","))
}

func (e EmbeddingVector) Value() (driver.Value, error) {
	return e.Stringify(), nil
}

func (e *EmbeddingVector) Scan(src any) error {
	if src == nil {
		return nil
	}

	*e = make([]float32, 0)
	str, ok := src.(string)
	if !ok {
		return errors.New("not a string")
	}
	str = strings.Trim(str, "[]")
	strs := strings.Split(str, ",")
	for _, s := range strs {
		f, err := strconv.ParseFloat(s, 32)
		if err != nil {
			return err
		}
		*e = append(*e, float32(f))
	}
	return nil
}
