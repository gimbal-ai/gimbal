# Development Environment

## Running the Control Plane

We currently only support running control plane in GKE (support for Minikube coming soon). To get a development version of the control plane up-and-running:

1. Add the dev cluster to your Kubeconfig:
```
gcloud container clusters get-credentials dev-cluster --zone us-west1-a --project gimlet-dev-0
```

2. Set the default repo for skaffold. This specifies the registry it should push images to.
```
skaffold config set default-repo us-docker.pkg.dev/gimlet-dev-0/gimlet-dev-docker-artifacts
```

3. Run skaffold to build and deploy the services in your namespace:
```
skaffold run -f skaffold/skaffold_controlplane.yaml -n <YOUR_USERNAME> -p dev
```
