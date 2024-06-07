# Gimlet API Documentation

This directory contains the API definitions and messages for the Gimlet platform. These APIs are considered internal.

We adopt a versioned protobuf naming convention. All packages should adhere to the following naming pattern: `<msg_pkg>/v<x>/<file_name>.proto`.
While each directory can contain multiple `.proto` files, they must all belong to the same package.

## Directory Structure

- **corepb/**:
  - Contains the foundational APIs and messages used throughout the Gimlet platform.
  - This includes messages exchanged between the edge and the control plane, as well as types that provide basic functionality.
- **python/**:
  - Contains the python SDK for interacting with the Gimlet platform.
