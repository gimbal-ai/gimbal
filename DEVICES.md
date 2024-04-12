# Setting up Devices
<!-- TOC -->

- [Setting up Devices](#setting-up-devices)
  - [Standalone devices ie james-jetson](#standalone-devices-ie-james-jetson)
  - [Simulated Devices](#simulated-devices)
  - [Fake Cameras](#fake-cameras)
    - [If deploying the GEM binary directly](#if-deploying-the-gem-binary-directly)
    - [If developing with skaffold](#if-developing-with-skaffold)

<!-- /TOC -->
```sh
# Ensure you set your KUBECONFIG to the right cluster ie
# export KUBECONFIG=~/nuc_k3s.yaml

export GML_CONTROLPLANE_ADDR=app.$USER.gimletlabs.dev:443
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

If you need to run the GEM on a specific node, you can use the `nodeSelector` option.
If you need to get past a taint, you can use the `tolerations` option.
For example, to run on `nuc-004` and get past the `recorded_video` taint, you can do:

```sh
export GML_CONTROLPLANE_ADDR=app.$USER.gimletlabs.dev:443
export GML_DEPLOY_KEY=<deploy-key> # replace with your deploy key

helm install my-gem-release oci://us-docker.pkg.dev/gimlet-dev-0/gimlet-dev-docker-artifacts/charts/gem -n gml \
--version 0.0.0-alpha1 \
--set "deployKey=${GML_DEPLOY_KEY}" \
--set "controlplaneAddr=${GML_CONTROLPLANE_ADDR}" \
--set-json "gem.hostNetwork=true" \
--set-json 'imagePullSecrets=[{"name": "gml-dev-artifact-registry"}]' \
--set-json "type.daemonset=true" \
--set-json 'gem.resources={"requests":{"squat.ai/video":1},"limits":{"squat.ai/video":1}}' \
--set-json 'tolerations=[{"effect": "NoSchedule", "key": "recorded_video", "operator": "Exists"}]' \
--set-json 'nodeSelector={"kubernetes.io/hostname": "nuc-004"}'
```

To use the GPUs:

```sh
export GML_CONTROLPLANE_ADDR=app.$USER.gimletlabs.dev:443
export GML_DEPLOY_KEY=<deploy-key> # replace with your deploy key

helm install my-gem-release oci://us-docker.pkg.dev/gimlet-dev-0/gimlet-dev-docker-artifacts/charts/gem -n gml \
--version 0.0.0-alpha1 \
--set "deployKey=${GML_DEPLOY_KEY}" \
--set "controlplaneAddr=${GML_CONTROLPLANE_ADDR}" \
--set-json "gem.hostNetwork=true" \
--set-json 'imagePullSecrets=[{"name": "gml-dev-artifact-registry"}]' \
--set-json "type.daemonset=true" \
--set-json 'images.gem.tag="intelgpu-dev-latest"' \
--set-json 'gem.extraEnv=[{"name": "LD_LIBRARY_PATH", "value": "/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/usr/local/lib"}]' \
--set-json 'gem.resources={"requests": {"squat.ai/video": 1, "gpu.intel.com/i915_monitoring": 1}, "limits": {"squat.ai/video": 1, "gpu.intel.com/i915_monitoring": 1}}'
```

## Standalone devices (ie james-jetson)

On your device

```sh
# Replace james-jetson with your device's hostname
scp ./src/gem/install.sh $USER@james-jetson:~/install.sh
```

Then after ssh-ing

```sh
export GML_CONTROLPLANE_ADDR=app.$USER.gimletlabs.dev:443
export GML_DEPLOY_KEY=<deploy-key> # replace with your deploy key
# Optionally set. This will tail the logs and exit when you ctrl-c.
# export GML_DEV_MODE="true"
./install.sh
```

This will be a common deployment pattern for devices that are not in a k8s cluster.

## Simulated Devices

Will deploy several GEMS with pre-recorded data to your k8s cluster.

```sh
export GML_CONTROLPLANE_ADDR=app.$USER.gimletlabs.dev:443
export GML_NUM_RECORDED=5 # How many to deploy per recording
export GML_MAX_INIT_DELAY_S=30.0 # Max num seconds of random delay to inject into GEM startup so that timeseries data is staggered. Must be floating point number for helm templating to correctly interpret this env var as a string.
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

### If deploying the GEM binary directly

Set `GML_VIDEO_FROM_FILE_OVERRIDE` or `--video_from_file_override` to the location of the
desired video file to use, and we will use that instead of a camera.

```sh
export GML_VIDEO_FROM_FILE_OVERRIDE=<path-to-video-file> # replace with your video file
bazel run -c opt //src/gem:gem -- --blob_store_dir=$HOME/.cache/gml
```

### If developing with skaffold

As a developer running with skaffold, you can use the `fake_camera` profile to quickly
deploy a GEM that uses a fake camera. The `GML_FAKE_VIDEO_GCS` lets you pick which
video (from Google Cloud Storage) to use.

```sh
export GML_FAKE_VIDEO_GCS=gs://gml-dev-videos/gimlets/automated-self-checkout/coca-cola.mp4 # replace with your video file (must be on GCS)
skaffold run -f ./skaffold/skaffold_gem.yaml -n $USER -p fake_camera
```

Fake camera devices will still run the real GEM and will be able to run models, but will use the pre-recorded video data.
