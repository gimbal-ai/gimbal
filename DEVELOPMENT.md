# Development Environment
>
> **Notice:**
> gcloud auth is configured to log you out every 24h. You must re-login every 24h
>
> ```sh
> gcloud auth login --update-adc
> ```

<!-- TOC -->

- [Development Environment](#development-environment)
  - [Pegboard Cluster](#pegboard-cluster)
    - [kubeconfig](#kubeconfig)
    - [Local Image Registry](#local-image-registry)
    - [GPU Requests](#gpu-requests)
    - [Priority Classes](#priority-classes)
  - [Running the Control Plane](#running-the-control-plane)
    - [Accessing the UI in the browser](#accessing-the-ui-in-the-browser)
  - [Running a GEM with a fake camera](#running-a-gem-with-a-fake-camera)
  - [Building the UI locally](#building-the-ui-locally)
    - [Accessing the development postgres DB](#accessing-the-development-postgres-db)
  - [Python Development](#python-development)
    - [Fast python image builds](#fast-python-image-builds)
    - [Python gazelle](#python-gazelle)

<!-- /TOC -->

## Pegboard Cluster

### kubeconfig

We have a k3s backed cluster of machines in the office named pegboard. The kubeconfig to access the cluster is stored in 1Password (or ask a coworker to send it to you).

To merge the kubeconfig for pegboard into your existing kubeconfig, you can run:

```sh
export EXTRA_CONFIG=$HOME/pegboard.yaml

cp $HOME/.kube/config $HOME/.kube/config_bkup
KUBECONFIG="$HOME/.kube/config:${EXTRA_CONFIG}" kubectl config view --flatten > merged.yaml
mv merged.yaml $HOME/.kube/config
```

### Local Image Registry

The cluster also runs a local image registry at [registry.pegboard.cluster.gimletlabs.dev](registry.pegboard.cluster.gimletlabs.dev). We prefer using this registry to both avoid egress costs on Google Artifact Registry and to keep our images private and local.

Once your kubectx is set to point to the pegboard cluster, you can run `skaffold config set default-repo registry.pegboard.cluster.gimletlabs.dev` to ensure that skaffold pushes images to the local registry whenever you deploy images to this cluster.

### GPU Requests

To run k8s containers that have access to the GPU, you need to request the GPU resource. You can do this by adding the following to your pod spec:

```yaml
nvidia.com/gpu: "1"
```

The GPUs are currently sliced into 2, so requesting 1 GPU will give you access to half of a GPU. Note that this doesn't enfore GPU memory limits and someone requesting half a GPU should ensure that their batch size is small enough to use only half the memory on the card.

You might also need to set the runtimeClassName to your Pod Spec for CUDA access to work. This is supposed to be the default class but setting it doesn't hurt.

```yaml
runtimeClassName: nvidia
```

### Priority Classes

The cluster has custom priority classes that can be used to prioritize workloads. The priority classes in order of descending priority are: `interactive-high, interactive-low, batch-high, batch-low`

There are also nonpreempting versions of every priority class. These are useful for running workloads that need eventual completion but should not preempt any existing workloads. Since nonpreempting workloads are expected to take longer to schedule in the queue, they are given a slightly higher priority than their preempting counterparts.

Workloads default to `batch-low` if no priority class is specified. Please set appropriate priority classes for your workloads as needed.

## Running the Control Plane

We currently only support running control plane in GKE (support for Minikube coming soon). To get a development version of the control plane up-and-running:

1. Add the dev cluster to your Kubeconfig:

    ```sh
    gcloud container clusters get-credentials dev-cluster --zone us-west1-a --project gimlet-dev-0
    ```

1. Set the default repo for skaffold. This specifies the registry it should push images to.

    ```sh
    skaffold config set default-repo us-docker.pkg.dev/gimlet-dev-0/gimlet-dev-docker-artifacts
    ```

1. Add `export SKAFFOLD_LABEL=skaffold.dev/run-id=PREVENT_REDEPLOY` to your .bashrc or .zshrc.

1. Configure docker to use gcloud for auth:

    ```sh
    gcloud auth configure-docker us-docker.pkg.dev
    ```

1. Run skaffold to deploy any dependencies to your namespace. You should only need to run this once unless you are configuring the dependencies. Unfortunately, the `helm dep update` is required due to [#8036](https://github.com/helm/helm/issues/8036), [#9903](https://github.com/helm/helm/issues/9903).

    ```sh
    helm dep update k8s/charts/controlplane_deps/
    skaffold run -f skaffold/skaffold_controlplane_deps.yaml -n <YOUR_USERNAME> -p dev
    ```

1. Run skaffold to build and deploy the services in your namespace:

    ```sh
    skaffold run -f skaffold/skaffold_controlplane.yaml -n <YOUR_USERNAME> -p dev
    ```

1. (optional) Setup your kubectl config to only look at your namespace:

    ```sh
    kubectl config set-context --current --namespace=<YOUR_USERNAME>
    ```

### Accessing the UI in the browser

We have ingress and LetsEncrypt signed certs automatically setup for the dev cluster. As long as you are on
tailscale, you should be able to just navigate to `<YOUR_USERNAME>.dev.app.gimletlabs.dev` via your browser
to access the UI.

## Running a GEM with a fake camera

If you need a GEM that sends data to your controlplane for testing purposes, you can skaffold a GEM binary in our dev cluster.

  ```sh
  export GML_DEPLOY_KEY=<DEPLOY_KEY>
  export GML_CONTROLPLANE_ADDR=$USER.dev.app.gimletlabs.dev
  skaffold run -f ./skaffold/skaffold_gem.yaml -n $USER
  ```

## Building the UI locally

1. Make sure you have envoy installed locally. Chef manages this on linux machines. If you are running macOS, you can use Homebrew.

    ```sh
    brew install envoy
    ```

1. Make sure you have certs for your machine. These are generated by tailscale. Our `run_proxy.sh` script expects these certs to be in your homedir, but you can override the location if needed by setting the SSL_CRT_PATH and SSL_KEY_PATH env vars.

    ```sh
    sudo tailscale cert ${HOSTNAME}.beluga-snapper.ts.net
    ```

1. Make the tailscale cert key readable by `sudo` (on Linux) or `admin` (on MacOS).

    ```sh
    sudo chgrp <ADMIN_GROUP> ${HOSTNAME}.beluga-snapper.ts.net.key
    sudo chmod g+r ${HOSTNAME}.beluga-snapper.ts.net.key
    ```

1. To point to a backend that isn't yours, set the `BACKEND` env var. We default to `${USER}.dev.app.gimletlabs.dev`

1. From the `src/ui` directory, run `pnpm dev`.

    ```sh
    pnpm dev
    ```

1. Access the dev UI at the envoy port with your host's FQDN. Since the envoy port is dynamically assigned, check the output from `pnpm dev` for your UI location. For eg `https://${HOSTNAME}.beluga-snapper.ts.net:8989`

### Accessing the development postgres DB

You can access the development postgres DB using the `access_dev_db.sh` helper script.

  ```sh
  ./scripts/access_dev_db.sh
  ```

Then you can run queries like so:

  ```sql
  SELECT * from fleets;
  ```

You should use transactions if you want to make changes to the DB:

  ```sql
  BEGIN;
  INSERT INTO fleets (name, org_id, created_at, updated_at) VALUES ('test', '09b4690d-5e92-437b-9401-41ee8edb3bdb', now(), now());

  -- If you want to rollback your changes, run:
  -- ROLLBACK;
  COMMIT;
  ```

## Python Development

In order to get functional IDE features in python, it's recommended to install the gml requirements in a virtualenv and point your IDE to that virtualenv.
These instructions use `pyenv` and `pyenv-virtualenv` to create the virtualenv, but any virtualenv tool should work.

1. Install `pyenv` and `pyenv-virtualenv`.
1. Create the virtualenv. With `pyenv`:

    ```bash
    pyenv virtualenv gml && pyenv local gml
    ```

1. Install the gml requirements:

    ```bash
    pip install -e .
    ```

### Fast python image builds

Python image builds can be very slow. The default `gml_py_image` rule will build a normal minimal python image.
This image will only include the relevant dependencies but can be very slow to build.
To reduce build time there is an additional `.fast` target for each `gml_py_image`.
This image will include all python pip packages in the image but is faster to build since it can better utilize the cache.
Keep in mind that this image will be larger than it needs to be and doesn't use bazel's hermetic python toolchain.
These bazel options can also help speed up the python image builds:

- `--bes_upload_mode=fully_async`
- `--noremote_upload_local_results`

### Python gazelle

When adding a new python dependency, we need multiple steps before gazelle can be run.

1. Add the dependency to `bazel/python/requirements.in`.

1. Run the `compile_pip_requirements` target to generate the `requirements_lock.txt` file.

    ```bash
    make update-python-requirements
    ```

1. Finally run the `gazelle_python_manifest` target to generate the `gazelle_python.yaml` file.

    ```bash
    make update-python-manifest
    ```

1. Finally run the `gazelle` target if needed to generate and update `BUILD.bazel` files.

    ```bash
    make gazelle
    ```

## C++ Development

In order to get proper C++ code analysis in your IDE, run this script first:

  ```bash
  ./scripts/gen_compilation_database.sh
  ```
