<!-- markdownlint-disable-next-line -->
![Gimbal](./.readme_assets/gimbal_logo_dark.png#gh-dark-mode-only)![Gimbal](./.readme_assets/gimbal_logo_light.png#gh-light-mode-only)

## Overview

Gimbal is a developer tool which provides MLOps for edge AI. By providing an abstraction layer for compatibility with diverse sensors, hardware, and underlying AI frameworks, Gimbal helps developers deploy and manage AI pipelines of model variants across a fleet of heterogeneous edge devices.

## Description

Gimbal simplifies the deployment and management of AI models and pipelines on a wide range of edge devices. The edge refers to a computing paradigm that brings data processing closer to the source of data generation, enabling faster, more efficient computing, especially in scenarios with limited connectivity or stringent real-time processing requirements.

Existing AI model development and deployment solutions are not designed to handle problems unique to the edge:

- Developers may be deploying to a fleet of heterogeneous devices, but models must be optimized for specific hardware and frameworks.
- Per device sensor and environmental differences may require heterogeneous models to achieve the best performance, however, deploying multiple variants of the model to your fleet of devices can be challenging.
- Models must be able to run in resource constrained environments, requiring low memory and CPU footprint


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

Gimbal offers a comprehensive suite of tools designed to streamline the deployment and manage the life-cycle of AI models across a diverse fleet of edge devices. Gimbal consists of the following major components.

**Gimbal Edge Module**: This module serves as a dynamic agent installed on each system node to streamline AI pipeline management. The edge module provides versioned model deployment, model observability, and last-mile optimization for ML models by utilizing HW optimized ML libraries such as TensorRT™ or OpenVino.

**Gimbal Control Plane**: A set of K8s services that are meant to be deployed on the cloud or other centralized infrastructure. These services provide infrastructure for model conversion, version control, and access to the observability data.

**Gimbal CLI**: Provides the primary means of interaction with Gimbal. The CLI allows for upload of new model pipelines, version management and basic status information.

The edge module is developed in modern C++ and connects with lower-level ML libraries and system primitives. The control plane and CLI are primarily written in Go and utilize cloud-native best practices for a microservices-based deployment.

## Development Process

1. **Clone the repo:**

    ```bash
    git clone git@github.com:gimbal-ai/gimbal.git
    ```

1. **Install dependencies:**

    There are few options to run the development environment for Gimbal. You can use the pre-built docker container which contains all the dependencies, use chef to provision an image, or manually install the dependencies.

    Using Docker:

    ```bash
    ./scripts/run_docker.sh
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

1. **Create a Fleet:**

  Go to the [hosted control plane](app.gimletlabs.ai) and create a fleet for your device.

1. **Create a Deploy Key:**

  Go to your fleet's settings and click "Create Deploy Key".

1. **Run GEM:**

    ```bash
    export GML_DEPLOY_KEY="<PUT KEY HERE>"
    bazel run -c opt //src/gem:gem
    ```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING).

## Roadmap

See [ROADMAP.md](ROADMAP).

## Governance

Gimbal is governed by a committee consisting of 6 members, with leadership provided by Gimlet Labs, Inc., and other founding members. Two seats are reserved for end users.

## License

This project is licensed under the [Apache License](LICENSE).
