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
	tx, err := db.Beginx()
	if err != nil {
		return tx, ErrFailedToCreateTx
	}

	md, _ := metadata.FromIncomingContext(ctx)
	idempotencyKeys := md.Get("x-idempotency-key")
	if len(idempotencyKeys) == 0 {
		return tx, ErrIdempotencyKeyMissing
	}
	idempotencyKey := idempotencyKeys[0]
	// Will fail if unique key already exists.
	_, err = tx.Exec(fmt.Sprintf("INSERT INTO %s_idempotency_keys (idempotency_key) VALUES ('%s')", svc, idempotencyKey))
	if err != nil {
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
