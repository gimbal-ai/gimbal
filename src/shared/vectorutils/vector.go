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
