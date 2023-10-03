# Development Environment

## Running the Control Plane

We currently only support running control plane in GKE (support for Minikube coming soon). To get a development version of the control plane up-and-running:

1. Add the dev cluster to your Kubeconfig:
```
gcloud container clusters get-credentials dev-cluster --zone us-west1-a --project gimlet-dev-0
```

2. Run skaffold to build and deploy the services in your namespace:
```
skaffold run -f skaffold/skaffold_controlplane.yaml -n <YOUR_USERNAME> -p dev
```
