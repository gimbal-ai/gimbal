# Setting up Devices
<!-- TOC -->

- [Setting up Devices](#setting-up-devices)
  - [NUC Cluster Kubernetes cluster for real devices](#nuc-cluster-kubernetes-cluster-for-real-devices)
  - [Standalone devices ie james-jetson](#standalone-devices-ie-james-jetson)
  - [Simulated Devices](#simulated-devices)
  - [Fake Cameras](#fake-cameras)

<!-- /TOC -->

This guide shows you how to setup the various devices types that we support.
You should be able to find the device type you are looking for and just follow the instructions.

## NUC Cluster (Kubernetes cluster for real devices)

Will deploy GEMs to each node in the kubernetes cluster.

```sh
# Ensure you set your KUBECONFIG to the right cluster ie
# export KUBECONFIG=~/nuc_k3s.yaml

export GML_CONTROLPLANE_ADDR=$USER.dev.app.gimletlabs.dev:443
export GML_DEPLOY_KEY=<deploy-key> # replace with your deploy key

helm install my-gem-release oci://us-docker.pkg.dev/gimlet-dev-0/gimlet-dev-docker-artifacts/charts/gem -n gml \
--version 0.0.0-alpha1 \
--set "deployKey=${GML_DEPLOY_KEY}" \
--set "controlplaneAddr=${GML_CONTROLPLANE_ADDR}" \
--set-json "gem.hostNetwork=true" \
--set-json 'imagePullSecrets=[{"name": "gml-dev-artifact-registry"}]' \
--set-json "type.daemonset=true" \
--set-json 'gem.resources={"requests":{"squat.ai/video":1},"limits":{"squat.ai/video":1}}'
```

## Standalone devices (ie james-jetson)

On your device

```sh
# Replace james-jetson with your device's hostname
scp ./src/gem/install.sh $USER@james-jetson:~/install.sh
```

Then after ssh-ing

```sh
export GML_CONTROLPLANE_ADDR=$USER.dev.app.gimletlabs.dev:443
export GML_DEPLOY_KEY=<deploy-key> # replace with your deploy key
# Optionally set. This will tail the logs and exit when you ctrl-c.
# export GML_DEV_MODE="true"
./install.sh
```

This will be a common deployment pattern for devices that are not in a k8s cluster.

## Simulated Devices

Will deploy several GEMS with pre-recorded data to your k8s cluster.

```sh
export GML_CONTROLPLANE_ADDR=$USER.dev.app.gimletlabs.dev:443
export GML_NUM_RECORDED=5 # How many to deploy per recording
export GML_API_KEY=<api-key> # replace with your api key
export GML_DEPLOY_KEY=<deploy-key> # replace with your deploy key
# These are the two recordings available as of time of writing. See `src/testutils/README.md`
# for instructions on how to create your own.
export GML_RECORDINGS="pk_vm_office_dance,boring_office_victory"
./scripts/deploy_recorded_gem_fleet.sh
```

This should be the fastest option for developing the UI/controlplane or for showing demos, but you won't be able to deploy arbitrary models.
If you'd like to record new data, see <https://github.com/gimletlabs/gimlet/blob/main/src/testutils/README.md>

## Fake Cameras

We only support this on the GKE dev-cluster, where we run fake cameras.

```sh
export GML_CONTROLPLANE_ADDR=$USER.dev.app.gimletlabs.dev:443
export GML_DEPLOY_KEY=<deploy-key> # replace with your deploy key

helm install my-gem-release oci://us-docker.pkg.dev/gimlet-dev-0/gimlet-dev-docker-artifacts/charts/gem -n $USER \
--version 0.0.0-alpha1 \
--set "deployKey=${GML_DEPLOY_KEY}" \
--set "controlplaneAddr=${GML_CONTROLPLANE_ADDR}" \
--set-json 'gem.resources={"requests":{"squat.ai/video":1},"limits":{"squat.ai/video":1}}'
```

Fake camera devices will still run the real GEM and will be able to run models, but will use the pre-recorded video data.