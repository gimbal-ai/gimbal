/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
 */

#include "src/gem/metrics/plugin/intelgpu/intel_gpu_metrics.h"

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

#include <chrono>
#include <map>

#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <sole.hpp>

#include "src/common/base/byte_utils.h"
#include "src/common/base/error.h"
#include "src/common/event/dispatcher.h"
#include "src/common/metrics/metrics_system.h"
#include "src/common/system/fdinfo.h"
#include "src/gem/metrics/core/shared_metric_names.h"

namespace gml::gem::metrics::intelgpu {

namespace {

std::string ZEResultEnumName(const ze_result_t result) {
  if (result == ZE_RESULT_SUCCESS) {
    return "ZE_RESULT_SUCCESS";
  } else if (result == ZE_RESULT_NOT_READY) {
    return "ZE_RESULT_NOT_READY";
  } else if (result == ZE_RESULT_ERROR_UNINITIALIZED) {
    return "ZE_RESULT_ERROR_UNINITIALIZED";
  } else if (result == ZE_RESULT_ERROR_DEVICE_LOST) {
    return "ZE_RESULT_ERROR_DEVICE_LOST";
  } else if (result == ZE_RESULT_ERROR_INVALID_ARGUMENT) {
    return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
  } else if (result == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
    return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
  } else if (result == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
    return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
  } else if (result == ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) {
    return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
  } else if (result == ZE_RESULT_ERROR_MODULE_LINK_FAILURE) {
    return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
  } else if (result == ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS) {
    return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
  } else if (result == ZE_RESULT_ERROR_NOT_AVAILABLE) {
    return "ZE_RESULT_ERROR_NOT_AVAILABLE";
  } else if (result == ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE) {
    return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
  } else if (result == ZE_RESULT_WARNING_DROPPED_DATA) {
    return "ZE_RESULT_WARNING_DROPPED_DATA";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_VERSION) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
  } else if (result == ZE_RESULT_ERROR_INVALID_NULL_HANDLE) {
    return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
  } else if (result == ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE) {
    return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
  } else if (result == ZE_RESULT_ERROR_INVALID_NULL_POINTER) {
    return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
  } else if (result == ZE_RESULT_ERROR_INVALID_SIZE) {
    return "ZE_RESULT_ERROR_INVALID_SIZE";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_SIZE) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
  } else if (result == ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT) {
    return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
  } else if (result == ZE_RESULT_ERROR_INVALID_ENUMERATION) {
    return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
  } else if (result == ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT) {
    return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
  } else if (result == ZE_RESULT_ERROR_INVALID_NATIVE_BINARY) {
    return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
  } else if (result == ZE_RESULT_ERROR_INVALID_GLOBAL_NAME) {
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_NAME) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_FUNCTION_NAME) {
    return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
  } else if (result == ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION) {
    return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
  } else if (result == ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION) {
    return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
  } else if (result == ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE) {
    return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
  } else if (result == ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED) {
    return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
  } else if (result == ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE) {
    return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
  } else if (result == ZE_RESULT_ERROR_OVERLAPPING_REGIONS) {
    return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
  } else if (result == ZE_RESULT_ERROR_UNKNOWN) {
    return "ZE_RESULT_ERROR_UNKNOWN";
  } else {
    return std::to_string(static_cast<int>(result));
  }
}

StatusOr<std::vector<ze_driver_handle_t>> CoreDrivers() {
  uint32_t driver_count = 0;
  ze_result_t status = zeDriverGet(&driver_count, nullptr);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get intel gpu drivers: $0", ZEResultEnumName(status));
  }

  if (driver_count == 0) {
    return std::vector<ze_driver_handle_t>{};
  }

  std::vector<ze_driver_handle_t> drivers(driver_count);
  status = zeDriverGet(&driver_count, drivers.data());
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get intel gpu drivers: $0", ZEResultEnumName(status));
  }
  drivers.resize(driver_count);
  return drivers;
}

StatusOr<std::vector<zes_driver_handle_t>> SysmanDrivers() {
  uint32_t driver_count = 0;
  ze_result_t status = zesDriverGet(&driver_count, nullptr);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get intel gpu drivers: $0", ZEResultEnumName(status));
  }

  if (driver_count == 0) {
    return std::vector<zes_driver_handle_t>{};
  }

  std::vector<zes_driver_handle_t> drivers(driver_count);
  status = zesDriverGet(&driver_count, drivers.data());
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get intel gpu drivers: $0", ZEResultEnumName(status));
  }
  drivers.resize(driver_count);
  return drivers;
}

StatusOr<std::vector<ze_device_handle_t>> DevicesForDriver(ze_driver_handle_t driver) {
  uint32_t device_count = 0;
  ze_result_t status = zeDeviceGet(driver, &device_count, nullptr);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device count: $0", ZEResultEnumName(status));
  }

  if (device_count == 0) {
    return std::vector<ze_device_handle_t>{};
  }

  std::vector<ze_device_handle_t> devices(device_count);
  status = zeDeviceGet(driver, &device_count, devices.data());
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get devices: $0", ZEResultEnumName(status));
  }
  devices.resize(device_count);
  return devices;
}

StatusOr<std::vector<zes_device_handle_t>> DevicesForSysmanDriver(zes_driver_handle_t driver) {
  uint32_t device_count = 0;
  ze_result_t status = zesDeviceGet(driver, &device_count, nullptr);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device count: $0", ZEResultEnumName(status));
  }

  if (device_count == 0) {
    return std::vector<zes_device_handle_t>{};
  }

  std::vector<zes_device_handle_t> devices(device_count);
  status = zesDeviceGet(driver, &device_count, devices.data());
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get devices: $0", ZEResultEnumName(status));
  }
  devices.resize(device_count);
  return devices;
}

StatusOr<ze_device_properties_t> GetDeviceProps(ze_device_handle_t device) {
  ze_device_properties_t props = {};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ze_result_t status = zeDeviceGetProperties(device, &props);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device properties: $0", ZEResultEnumName(status));
  }
  return props;
}

StatusOr<zes_device_properties_t> GetSysmanDeviceProps(zes_device_handle_t device) {
  zes_device_properties_t props = {};
  props.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ze_result_t status = zesDeviceGetProperties(device, &props);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device properties: $0", ZEResultEnumName(status));
  }
  return props;
}

StatusOr<std::vector<ze_device_memory_properties_t>> GetDeviceMemProps(ze_device_handle_t device) {
  uint32_t props_count = 0;
  ze_result_t status = zeDeviceGetMemoryProperties(device, &props_count, nullptr);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device memory properties: $0", ZEResultEnumName(status));
  }

  if (props_count == 0) {
    return std::vector<ze_device_memory_properties_t>{};
  }

  std::vector<ze_device_memory_properties_t> props(props_count);
  for (auto& prop : props) {
    prop.stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
  }
  status = zeDeviceGetMemoryProperties(device, &props_count, props.data());
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device memory properties: $0", ZEResultEnumName(status));
  }

  props.resize(props_count);
  return props;
}

sole::uuid ZEUUIDToSole(ze_device_uuid_t uuid) {
  return {
      utils::LEndianBytesToInt<uint64_t>(std::string_view(reinterpret_cast<char*>(&uuid.id[0]), 8)),
      utils::LEndianBytesToInt<uint64_t>(std::string_view(reinterpret_cast<char*>(&uuid.id[8]), 8)),
  };
}

StatusOr<std::vector<zes_engine_handle_t>> GetDeviceEngines(zes_device_handle_t device) {
  uint32_t count = 0;
  ze_result_t status = zesDeviceEnumEngineGroups(device, &count, nullptr);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device engines: $0", ZEResultEnumName(status));
  }

  if (count == 0) {
    return std::vector<zes_engine_handle_t>{};
  }

  std::vector<zes_engine_handle_t> engines(count);
  status = zesDeviceEnumEngineGroups(device, &count, engines.data());
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get device engines: $0", ZEResultEnumName(status));
  }
  engines.resize(count);
  return engines;
}

StatusOr<zes_engine_properties_t> GetEngineProps(zes_engine_handle_t engine) {
  zes_engine_properties_t props = {};
  props.stype = ZES_STRUCTURE_TYPE_ENGINE_PROPERTIES;
  ze_result_t status = zesEngineGetProperties(engine, &props);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get engine properties: $0", ZEResultEnumName(status));
  }
  return props;
}

StatusOr<zes_engine_stats_t> GetEngineStats(zes_engine_handle_t engine) {
  zes_engine_stats_t stats = {};
  ze_result_t status = zesEngineGetActivity(engine, &stats);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get engine stats: $0", ZEResultEnumName(status));
  }
  return stats;
}

StatusOr<zes_pci_properties_t> GetDevicePCIProperties(zes_device_handle_t device) {
  zes_pci_properties_t props = {};
  props.stype = ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
  ze_result_t status = zesDevicePciGetProperties(device, &props);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to get PCI properties: $0", ZEResultEnumName(status));
  }
  return props;
}

template <typename T>
auto GetObservableResult(opentelemetry::metrics::ObserverResult& observer) {
  return std::get<std::shared_ptr<opentelemetry::metrics::ObserverResultT<T>>>(observer);
}

}  // namespace

IntelGPUMetrics::IntelGPUMetrics(gml::metrics::MetricsSystem* metrics_system,
                                 gml::event::Dispatcher*)
    : core::Scraper(metrics_system) {
  auto gml_meter = metrics_system_->GetMeterProvider()->GetMeter("gml");

  system_utilization_counter_ =
      gml_meter->CreateDoubleObservableCounter(core::kGPUUtilizationSystemCounterName);
  system_utilization_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto gpu_metrics = static_cast<IntelGPUMetrics*>(parent);
        absl::base_internal::SpinLockHolder lock(&gpu_metrics->metrics_lock_);

        for (const auto& [device_id, metrics] : gpu_metrics->device_metrics_) {
          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(metrics.system_counter_ns) / 1E9, {
                                                                        {"gpu_id", device_id},
                                                                    });
        }
      },
      this);
  gem_utilization_counter_ =
      gml_meter->CreateDoubleObservableCounter(core::kGPUUtilizationGEMCounterName);
  gem_utilization_counter_->AddCallback(
      [](auto observer, void* parent) {
        auto gpu_metrics = static_cast<IntelGPUMetrics*>(parent);
        absl::base_internal::SpinLockHolder lock(&gpu_metrics->metrics_lock_);

        for (const auto& [device_id, metrics] : gpu_metrics->device_metrics_) {
          GetObservableResult<double>(observer)->Observe(
              static_cast<double>(metrics.gem_counter_ns) / 1E9, {
                                                                     {"gpu_id", device_id},
                                                                 });
        }
      },
      this);

  system_memory_size_gauge_ = gml_meter->CreateInt64Gauge(core::kGPUMemorySystemSizeGaugeName);
  start_time_ = std::chrono::steady_clock::now();
}

void IntelGPUMetrics::Scrape() {
  auto s = ScrapeWithError();
  if (!s.ok()) {
    LOG(ERROR) << "Failed to scrape intel gpu metrics: " << s.msg();
  }
}

Status IntelGPUMetrics::ScrapeWithError() {
  ze_result_t status = zeInit(0);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to init level_zero core API: $0", ZEResultEnumName(status));
  }
  status = zesInit(0);
  if (status != ZE_RESULT_SUCCESS) {
    return error::Internal("failed to init level_zero sysman API: $0", ZEResultEnumName(status));
  }

  absl::base_internal::SpinLockHolder lock(&metrics_lock_);

  GML_ASSIGN_OR_RETURN(auto drivers, CoreDrivers());
  absl::flat_hash_map<sole::uuid, uint32_t> uuid_to_device_id;
  std::map<uint32_t, ze_device_handle_t> gpu_devices;
  // We get a handle for each device that can be used with the core APIs.
  for (auto driver : drivers) {
    GML_ASSIGN_OR_RETURN(auto devices, DevicesForDriver(driver));
    for (auto device : devices) {
      GML_ASSIGN_OR_RETURN(auto device_properties, GetDeviceProps(device));
      if (device_properties.type == ZE_DEVICE_TYPE_GPU) {
        sole::uuid uuid = ZEUUIDToSole(device_properties.uuid);
        uuid_to_device_id.emplace(uuid, device_properties.deviceId);
        gpu_devices.emplace(device_properties.deviceId, device);
      }
    }
  }

  GML_ASSIGN_OR_RETURN(auto sysman_drivers, SysmanDrivers());
  std::map<uint32_t, zes_device_handle_t> sysman_devices;
  // We get a handle for each device that can be used with the "sysman" APIs.
  for (auto driver : sysman_drivers) {
    GML_ASSIGN_OR_RETURN(auto devices, DevicesForSysmanDriver(driver));
    for (auto device : devices) {
      GML_ASSIGN_OR_RETURN(auto device_properties, GetSysmanDeviceProps(device));
      sole::uuid uuid = ZEUUIDToSole(device_properties.core.uuid);
      if (!uuid_to_device_id.contains(uuid)) {
        continue;
      }
      // device_properties.core.deviceId is not populated on the sysman device properties.
      // So we need to lookup the UUID to get the device ID from the core device properties.
      auto device_id = uuid_to_device_id[uuid];
      sysman_devices.emplace(device_id, device);
    }
  }

  // From each core handle we get the device's total memory size.
  for (auto [device_id, device] : gpu_devices) {
    GML_ASSIGN_OR_RETURN(auto memory_properties, GetDeviceMemProps(device));
    uint64_t total_memory = 0;
    for (auto prop : memory_properties) {
      total_memory += prop.totalSize;
    }
    VLOG(1) << absl::Substitute("Device $0: Total memory $1MB", device_id,
                                static_cast<float>(total_memory) / 1024.0 / 1024.0);
    system_memory_size_gauge_->Record(total_memory,
                                      {
                                          {"gpu_id", absl::StrCat(device_id)},
                                      },
                                      {});
  }

  absl::flat_hash_map<std::string, std::string> pci_addr_to_device_id;

  // From each sysman handle, we get the device's active time across all processes.
  for (auto [device_id, device] : sysman_devices) {
    GML_ASSIGN_OR_RETURN(auto engines, GetDeviceEngines(device));
    std::multimap<zes_engine_group_t, zes_engine_handle_t> engine_by_type;
    for (auto engine : engines) {
      GML_ASSIGN_OR_RETURN(auto props, GetEngineProps(engine));
      engine_by_type.emplace(props.type, engine);
    }

    // Some intel cards report GEMM usage under "COMPUTE" and some report it under "RENDER".
    // In both cases, some cards have multiple "engines", in which case the underlying hardware
    // usage is reported in `ZES_ENGINE_GROUP_*_ALL`. Some cards don't report that engine and
    // instead only report a single `ZES_ENGINE_GROUP_*_SINGLE`. So we first look for the `ALL`
    // engine group and fallback to the `SINGLE` group.
    zes_engine_handle_t relevant_engine;
    if (engine_by_type.count(ZES_ENGINE_GROUP_COMPUTE_ALL) > 0) {
      relevant_engine = engine_by_type.find(ZES_ENGINE_GROUP_COMPUTE_ALL)->second;
    } else if (engine_by_type.count(ZES_ENGINE_GROUP_COMPUTE_SINGLE) == 1) {
      relevant_engine = engine_by_type.find(ZES_ENGINE_GROUP_COMPUTE_SINGLE)->second;
    } else if (engine_by_type.count(ZES_ENGINE_GROUP_RENDER_ALL) > 0) {
      relevant_engine = engine_by_type.find(ZES_ENGINE_GROUP_RENDER_ALL)->second;
    } else if (engine_by_type.count(ZES_ENGINE_GROUP_RENDER_SINGLE) == 1) {
      relevant_engine = engine_by_type.find(ZES_ENGINE_GROUP_RENDER_SINGLE)->second;
    } else {
      return error::Internal("could not find engine to record usage stats");
    }
    GML_ASSIGN_OR_RETURN(auto stats, GetEngineStats(relevant_engine));

    auto device_id_str = absl::StrFormat("0x%x", device_id);

    device_metrics_[device_id_str] = DeviceMetrics{
        // The API doesn't guarantee the units for activeTime. Experiments suggest that its units
        // are microseconds, but we might need to try it out on a larger variety of hardware to make
        // sure.
        .system_counter_ns = 1000 * stats.activeTime,
        .gem_counter_ns = 0,
    };

    // Get the PCI bus address for the device so that we can map fdinfo statistics to the device id
    // from level zero.
    GML_ASSIGN_OR_RETURN(auto pci_props, GetDevicePCIProperties(device));
    auto pci_addr =
        absl::StrFormat("%04d:%02d:%02d.%01d", pci_props.address.domain, pci_props.address.bus,
                        pci_props.address.device, pci_props.address.function);
    pci_addr_to_device_id.emplace(pci_addr, device_id_str);
  }

  std::vector<system::FDInfo> fdinfos;
  GML_RETURN_IF_ERROR(proc_parser_.ParseProcPIDFDInfo("self", &fdinfos));

  for (const auto& fdinfo : fdinfos) {
    if (!fdinfo.ext || fdinfo.ext->Type() != system::FDInfoExtension::FDINFO_TYPE_DRM) {
      continue;
    }

    const auto* drm_info = static_cast<system::DRMFDInfo*>(fdinfo.ext.get());

    auto it = pci_addr_to_device_id.find(drm_info->pdev());
    if (it == pci_addr_to_device_id.end()) {
      continue;
    }
    auto device_id = it->second;

    // TODO(james): check if discrete graphics cards also report usage under the "render" engine.
    if (!drm_info->engines().contains("render")) {
      continue;
    }
    // Add the active time across all open file descriptors with the same PCI address.
    device_metrics_[device_id].gem_counter_ns += drm_info->engines().find("render")->second.busy_ns;
  }

  for (const auto& [device_id, metrics] : device_metrics_) {
    VLOG(1) << absl::Substitute("Device $0: Engine active for $1. GEM active for $2", device_id,
                                metrics.system_counter_ns, metrics.gem_counter_ns);
  }

  return Status::OK();
}

}  // namespace gml::gem::metrics::intelgpu
