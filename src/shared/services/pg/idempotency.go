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

package pg

import (
	"context"
	"errors"
	"fmt"

	// This is required to get the "pgx" driver.
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

var (
	ErrFailedToCreateTx      = errors.New("failed to begin new transaction")
	ErrIdempotencyTxFailed   = errors.New("transaction not idempotent")
	ErrIdempotencyKeyMissing = errors.New("idempotency key missing from context")
)

// CreateIdempotentTx creates a transaction for a query that will only run if the idempotency key does not yet exist.
func CreateIdempotentTx(ctx context.Context, db *sqlx.DB, svc string) (*sqlx.Tx, error) {
	tx, err := db.BeginTxx(ctx, nil)
	if err != nil {
		return tx, ErrFailedToCreateTx
	}

	md, _ := metadata.FromIncomingContext(ctx)
	idempotencyKeys := md.Get("x-idempotency-key")
	if len(idempotencyKeys) == 0 {
		tx.Rollback()
		return tx, ErrIdempotencyKeyMissing
	}
	idempotencyKey := idempotencyKeys[0]
	// Will fail if unique key already exists.
	_, err = tx.ExecContext(ctx, fmt.Sprintf("INSERT INTO %s_idempotency_key (idempotency_key) VALUES ($1)", svc), idempotencyKey)
	if err != nil {
		tx.Rollback()
		return tx, ErrIdempotencyTxFailed
	}
	return tx, nil
}

// ErrorToGRPCError returns the gRPC status error from an idempotent transaction failure.
func ErrorToGRPCError(err error) error {
	switch err {
	case ErrFailedToCreateTx:
		return status.Error(codes.Internal, "Failed to create transaction in db")
	case ErrIdempotencyTxFailed:
		return status.Error(codes.Aborted, "Failed to insert idempotent key, likely duplicate request")
	case ErrIdempotencyKeyMissing:
		return status.Error(codes.InvalidArgument, "Missing idempotency key in request context")
	default:
		return status.Error(codes.Internal, "Unknown error")
	}
}
