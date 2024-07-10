# Gimbal

## Overview

Gimbal is a developer tool which provides MLOps for edge AI. By providing an abstraction layer for compatibility with diverse sensors, hardware, and underlying AI frameworks, Gimbal helps developers deploy and manage AI pipelines of model variants across a fleet of heterogeneous edge devices.

Under :construction:!

## Contact

- Zain Asgar: <zasgar@gimletlabs.ai>
- Michelle Nguyen: <michelle@gimletlabs.ai>

## Description

Gimbal simplifies the deployment and management of AI models and pipelines on a wide range of edge devices. The edge refers to a computing paradigm that brings data processing closer to the source of data generation, enabling faster, more efficient computing, especially in scenarios with limited connectivity or stringent real-time processing requirements.

### Goals

- Support end-to-end deployment of pipelines on the edge.
- Provide a hardware abstraction layer for heterogeneous environments.
- Run in resource-constrained environments with minimal CPU/memory footprint.
- Support deployment of heterogeneous models.

### Non-Goals

- Deployment in centralized server environments.
- Replacing existing machine learning libraries.
- Management of the underlying system (e.g., provisioning nodes, installing OS).

## Architecture & Implementation

The control plane and CLI are primarily written in Go and utilize cloud-native best practices for a microservices-based deployment. The edge module is developed in modern C++ and connects with lower-level ML libraries and system primitives.

## Directory Structure

## Development Process

1. **Clone the repo:**

    ```bash
    git clone git@github.com:gimbal-ai/gimbal.git
    ```

1. **Install dependencies:**

    There are few options to run the development environment for Gimbal. You can use the pre-built docker container which contains all the dependencies, use chef to provision an image, or manually install the dependencies.

    Using Docker:

    ```bash
    docker run -it us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts/dev_image_with_extras:202407091449
    ```

    Using Chef:

    ```bash
    cd infra/chef
    sudo chef-solo -c solo.rb -j node_workstation.json
    ```

    Install Deps Manually:
    **Coming Soon**

1. **Running Tests:**

    To run all the tests you can simply run:

    ```bash
    bazel test //...
    ```

1. **Run GEM:**

    ```bash
    export GML_DEPLOY_KEY="<PUT KEY HERE>"
    bazel run -c opt //src/gem:gem
    ```

1. **Contributing:**

    - Please discuss any contributions with out team. The repo is currently not open for contributions until our final repo release (planned for mid-August).

## Roadmap

- Clean up the open source repo, accept external contribution, etc.
- Deploy Gem's as K8s native constructs.
- Expand support for additional sensor/model modalities beyond vision.
- Extend OS support to include Windows.
- Add support for other hardware accelerators with community contributions.

## Governance

Gimbal is governed by a committee consisting of 6 members, with leadership provided by Gimlet Labs, Inc., and other founding members. Two seats are reserved for end users.

## License

This project is licensed under the [Apache License](LICENSE).
