# Copyright 2018- The Pixie Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

ENV["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
ENV["CLOUDSDK_INSTALL_DIR"] = "/opt"
ENV["PATH"] = "/opt/google-cloud-sdk/bin:#{ENV['PATH']}"

include_recipe "gml_dev_extras::gperftools"

pkg_list = [
  "cmake",
  "emacs",
  "graphviz",
  "jq",
  "vim",
  "gpg",
  "pinentry-tty",
]

if platform_family?("debian")
  apt_package pkg_list + ["zsh"] do
    action :upgrade
  end
end

if platform_family?("mac_os_x")
  homebrew_package pkg_list
end

remote_file "/usr/local/share/zsh/site-functions/_bazel" do
  source node["bazel"]["zsh_completions"]
  mode 0644
  checksum node["bazel"]["zcomp_sha256"]
end

common_remote_bin "kubectl"
common_remote_tar_bin "kustomize"
common_remote_tar_bin "ksops"
common_remote_bin "minikube"
common_remote_bin "skaffold"
common_remote_bin "sops"
common_remote_bin "yq"
common_remote_bin "envoy"

common_remote_tar_bin "gh" do
  tool_loc "bin/gh"
  strip_components 1
end

common_remote_tar_bin "helm" do
  strip_components 1
end

execute "install gcloud" do
  command "curl https://sdk.cloud.google.com | bash"
  creates "/opt/google-cloud-sdk"
  action :run
end

execute "update gcloud" do
  command "gcloud components update"
  action :run
end

execute "install components" do
  command "gcloud components install beta gke-gcloud-auth-plugin docker-credential-gcr"
  action :run
end

directory "/opt/google-cloud-sdk/.install/.backup" do
  action :delete
  recursive true
end

execute "remove gcloud pycache" do
  action :run
  cwd "/opt/google-cloud-sdk"
  command "find . -regex '.*/__pycache__' -exec rm -r {} +"
end

execute "configure docker-credential-gcr" do
  command "docker-credential-gcr configure-docker"
  action :run
end

common_remote_zip 'packer'

common_remote_zip 'terraform'

directory "/usr/local/lib/docker/cli-plugins" do
  action :create
  recursive true
  owner node["owner"]
  group node["group"]
  mode "0755"
end

remote_file "/usr/local/lib/docker/cli-plugins/docker-buildx" do
  source node["docker-buildx"]["download_path"]
  mode "0755"
  checksum node["docker-buildx"]["sha256"]
end

execute "install go binaries" do
  ENV["GOPATH"] = "/opt/gml_dev/gopath"
  command %(go install github.com/bazelbuild/buildtools/buildifier@latest && \
            go install github.com/bazelbuild/buildtools/buildozer@latest && \
            go install github.com/derailed/k9s@v0.27.4 && \
            go clean -modcache && \
            go clean -cache)
end
