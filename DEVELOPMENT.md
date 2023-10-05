# Development Environment
>
> **Notice:**
> gcloud auth is configured to log you out every 24h. You must re-login every 24h
>
> ```sh
> gcloud auth login
> ```

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

1. Configure docker to use gcloud for auth:

    ```sh
    gcloud auth configure-docker us-docker.pkg.dev
    ```

1. Run skaffold to build and deploy the services in your namespace:

    ```sh
    skaffold run -f skaffold/skaffold_controlplane.yaml -n <YOUR_USERNAME> -p dev
    ```

### Accessing the UI in the browser

We have ingress and LetsEncrypt signed certs automatically setup for the dev cluster. As long as you are on
tailscale, you should be able to just navigate to `<YOUR_USERNAME>.dev.app.gimletlabs.dev` via your browser
to access the UI.

## Building the UI locally

1. Set the env vars:

    ```sh
    export DEV_API_SERVICE="https://<YOUR_NAMESPACE>.dev.app.gimletlabs.dev"
    export AUTH0_CLIENT_ID='f66GcpkIKEliCvvPoejx3Y6mlmYiM8p6'
    export AUTH0_ISSUER_BASE_URL='gimlet-dev.us.auth0.com'
    export GML_DISABLE_SSL=true
    ```

1. From the `src/ui` directory, run:

    ```sh
    PORT=<YOUR_PORT> pnpm dev
    ```
