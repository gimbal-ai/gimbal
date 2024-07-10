# Copyright 2023- Gimlet Labs, Inc.
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

unified_mode true
provides :remote_deb

default_action :create

property :name, String, name_property: true

action :create do
  deb_path =  "/tmp/#{new_resource.name}.deb"

  remote_file deb_path do
    source node[new_resource.name]["download_path"]
    mode "0644"
    checksum node[new_resource.name]["sha256"]
  end

  dpkg_package "#{new_resource.name}" do
    source deb_path
    action :install
  end

  file deb_path do
    action :delete
  end
end
