#include <sycl/sycl.hpp>
#include <stdio.h>
#include <vector>
#include <string>
#include <array>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Suppress deprecation warnings: we intentionally query deprecated properties
// to provide a complete device property report.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

extern "C" EXPORT
void print_all_device_properties() {
    auto platforms = sycl::platform::get_platforms();
    int device_idx = 0;

    for (auto &platform : platforms) {
        auto devices = platform.get_devices(sycl::info::device_type::gpu);
        for (auto &dev : devices) {
            printf("=== XPU Device %d ===\n", device_idx);

            // ----------------------------------------------------------------
            // Platform Info
            // ----------------------------------------------------------------
            std::string plat_name    = platform.get_info<sycl::info::platform::name>();
            std::string plat_vendor  = platform.get_info<sycl::info::platform::vendor>();
            std::string plat_version = platform.get_info<sycl::info::platform::version>();
            printf("  platform_name (platform name): %s\n",    plat_name.c_str());
            printf("  platform_vendor (platform vendor): %s\n", plat_vendor.c_str());
            printf("  platform_version (platform version): %s\n", plat_version.c_str());

            // ----------------------------------------------------------------
            // General Device Info
            // ----------------------------------------------------------------
            auto dev_type = dev.get_info<sycl::info::device::device_type>();
            const char* dev_type_str = "unknown";
            if      (dev_type == sycl::info::device_type::cpu)         dev_type_str = "cpu";
            else if (dev_type == sycl::info::device_type::gpu)         dev_type_str = "gpu";
            else if (dev_type == sycl::info::device_type::accelerator) dev_type_str = "accelerator";
            else if (dev_type == sycl::info::device_type::custom)      dev_type_str = "custom";
            else if (dev_type == sycl::info::device_type::automatic)   dev_type_str = "automatic";
            else if (dev_type == sycl::info::device_type::host)        dev_type_str = "host";
            printf("  device_type (device type): %s\n", dev_type_str);

            auto vendor_id = dev.get_info<sycl::info::device::vendor_id>();
            printf("  vendor_id (unique vendor ID): %u\n", vendor_id);

            std::string dev_name    = dev.get_info<sycl::info::device::name>();
            std::string dev_vendor  = dev.get_info<sycl::info::device::vendor>();
            std::string driver_ver  = dev.get_info<sycl::info::device::driver_version>();
            std::string dev_version = dev.get_info<sycl::info::device::version>();
            std::string backend_ver = dev.get_info<sycl::info::device::backend_version>();
            printf("  name (device name): %s\n",                         dev_name.c_str());
            printf("  vendor (device vendor): %s\n",                     dev_vendor.c_str());
            printf("  driver_version (driver version): %s\n",            driver_ver.c_str());
            printf("  version (device version string): %s\n",            dev_version.c_str());
            printf("  backend_version (backend-specific version): %s\n", backend_ver.c_str());
            try {
                auto profile = dev.get_info<sycl::info::device::profile>();
                printf("  profile (device profile, deprecated): %s\n", profile.c_str());
            } catch (...) { printf("  profile (device profile, deprecated): n/a\n"); }

            // ----------------------------------------------------------------
            // Compute
            // ----------------------------------------------------------------
            auto max_compute_units   = dev.get_info<sycl::info::device::max_compute_units>();
            auto max_work_group_size = dev.get_info<sycl::info::device::max_work_group_size>();
            auto max_work_item_dims  = dev.get_info<sycl::info::device::max_work_item_dimensions>();
            auto max_work_item_sizes = dev.get_info<sycl::info::device::max_work_item_sizes<3>>();
            auto max_num_sub_groups  = dev.get_info<sycl::info::device::max_num_sub_groups>();
            auto sub_group_sizes     = dev.get_info<sycl::info::device::sub_group_sizes>();
            auto sub_group_ifp       = dev.get_info<sycl::info::device::sub_group_independent_forward_progress>();
            auto address_bits        = dev.get_info<sycl::info::device::address_bits>();
            printf("  max_compute_units (max number of parallel compute units): %u\n", max_compute_units);
            printf("  max_work_group_size (max total work-items in a work-group): %zu\n", max_work_group_size);
            printf("  max_work_item_dimensions (max dimensions for work-items): %u\n", max_work_item_dims);
            printf("  max_work_item_sizes[0] (max work-items in dimension 0): %zu\n", max_work_item_sizes[0]);
            printf("  max_work_item_sizes[1] (max work-items in dimension 1): %zu\n", max_work_item_sizes[1]);
            printf("  max_work_item_sizes[2] (max work-items in dimension 2): %zu\n", max_work_item_sizes[2]);
            printf("  max_num_sub_groups (max number of sub-groups in a work-group): %u\n", max_num_sub_groups);
            printf("  sub_group_sizes (supported sub-group sizes):");
            for (auto sz : sub_group_sizes) printf(" %zu", sz);
            printf("\n");
            printf("  sub_group_independent_forward_progress (sub-groups make independent forward progress): %d\n", (int)sub_group_ifp);
            printf("  address_bits (default compute device address space size in bits): %u\n", address_bits);

            // -- Execution capabilities --
            try {
                auto exec_caps = dev.get_info<sycl::info::device::execution_capabilities>();
                printf("  execution_capabilities (supported execution modes):");
                for (auto cap : exec_caps) {
                    if      (cap == sycl::info::execution_capability::exec_kernel)        printf(" exec_kernel");
                    else if (cap == sycl::info::execution_capability::exec_native_kernel) printf(" exec_native_kernel");
                }
                printf("\n");
            } catch (...) { printf("  execution_capabilities (supported execution modes): n/a\n"); }

            // ----------------------------------------------------------------
            // Clock
            // ----------------------------------------------------------------
            auto max_clock_frequency = dev.get_info<sycl::info::device::max_clock_frequency>();
            printf("  max_clock_frequency (max clock frequency in MHz): %u\n", max_clock_frequency);

            // ----------------------------------------------------------------
            // Memory
            // ----------------------------------------------------------------
            auto global_mem_size            = dev.get_info<sycl::info::device::global_mem_size>();
            auto global_mem_cache_size      = dev.get_info<sycl::info::device::global_mem_cache_size>();
            auto global_mem_cache_line_size = dev.get_info<sycl::info::device::global_mem_cache_line_size>();
            auto global_mem_cache_type      = dev.get_info<sycl::info::device::global_mem_cache_type>();
            printf("  global_mem_size (size of global memory in bytes): %llu\n", (unsigned long long)global_mem_size);
            printf("  global_mem_cache_size (size of global memory cache in bytes): %llu\n", (unsigned long long)global_mem_cache_size);
            printf("  global_mem_cache_line_size (global memory cache line size in bytes): %u\n", global_mem_cache_line_size);
            const char* cache_type_str = "none";
            if      (global_mem_cache_type == sycl::info::global_mem_cache_type::read_only)  cache_type_str = "read_only";
            else if (global_mem_cache_type == sycl::info::global_mem_cache_type::read_write) cache_type_str = "read_write";
            printf("  global_mem_cache_type (type of global memory cache): %s\n", cache_type_str);

            auto local_mem_size = dev.get_info<sycl::info::device::local_mem_size>();
            auto local_mem_type = dev.get_info<sycl::info::device::local_mem_type>();
            printf("  local_mem_size (size of local memory in bytes): %llu\n", (unsigned long long)local_mem_size);
            printf("  local_mem_type (type of local memory): %s\n",
                   local_mem_type == sycl::info::local_mem_type::local  ? "local"  :
                   local_mem_type == sycl::info::local_mem_type::global ? "global" : "none");

            auto max_mem_alloc_size    = dev.get_info<sycl::info::device::max_mem_alloc_size>();
            auto mem_base_addr_align   = dev.get_info<sycl::info::device::mem_base_addr_align>();
            printf("  max_mem_alloc_size (max size of memory object allocation in bytes): %llu\n", (unsigned long long)max_mem_alloc_size);
            printf("  mem_base_addr_align (minimum alignment for any data type in bits): %u\n", mem_base_addr_align);
            try {
                auto max_constant_buf_size = dev.get_info<sycl::info::device::max_constant_buffer_size>();
                printf("  max_constant_buffer_size (max constant buffer size in bytes, deprecated): %llu\n", (unsigned long long)max_constant_buf_size);
            } catch (...) { printf("  max_constant_buffer_size (deprecated): n/a\n"); }
            try {
                auto max_constant_args = dev.get_info<sycl::info::device::max_constant_args>();
                printf("  max_constant_args (max number of constant arguments to a kernel, deprecated): %u\n", max_constant_args);
            } catch (...) { printf("  max_constant_args (deprecated): n/a\n"); }

            // ----------------------------------------------------------------
            // Image Support (deprecated query; use aspect::ext_intel_legacy_image)
            // ----------------------------------------------------------------
            auto image_support = dev.get_info<sycl::info::device::image_support>();
            printf("  image_support (device supports SYCL 1.2.1 images, deprecated): %d\n", (int)image_support);

            if (image_support) {
                auto max_read_image_args   = dev.get_info<sycl::info::device::max_read_image_args>();
                auto max_write_image_args  = dev.get_info<sycl::info::device::max_write_image_args>();
                auto image2d_max_width     = dev.get_info<sycl::info::device::image2d_max_width>();
                auto image2d_max_height    = dev.get_info<sycl::info::device::image2d_max_height>();
                auto image3d_max_width     = dev.get_info<sycl::info::device::image3d_max_width>();
                auto image3d_max_height    = dev.get_info<sycl::info::device::image3d_max_height>();
                auto image3d_max_depth     = dev.get_info<sycl::info::device::image3d_max_depth>();
                auto image_max_buffer_size = dev.get_info<sycl::info::device::image_max_buffer_size>();
                auto image_max_array_size  = dev.get_info<sycl::info::device::image_max_array_size>();
                auto max_samplers          = dev.get_info<sycl::info::device::max_samplers>();
                printf("  max_read_image_args (max simultaneous image read arguments): %u\n", max_read_image_args);
                printf("  max_write_image_args (max simultaneous image write arguments): %u\n", max_write_image_args);
                printf("  image2d_max_width (max width of 2D image in pixels): %zu\n", image2d_max_width);
                printf("  image2d_max_height (max height of 2D image in pixels): %zu\n", image2d_max_height);
                printf("  image3d_max_width (max width of 3D image in pixels): %zu\n", image3d_max_width);
                printf("  image3d_max_height (max height of 3D image in pixels): %zu\n", image3d_max_height);
                printf("  image3d_max_depth (max depth of 3D image in pixels): %zu\n", image3d_max_depth);
                printf("  image_max_buffer_size (max buffer size for 1D image in pixels): %zu\n", image_max_buffer_size);
                printf("  image_max_array_size (max images in 1D/2D image array, deprecated): %zu\n", image_max_array_size);
                printf("  max_samplers (max number of samplers): %u\n", max_samplers);
            }

            // ----------------------------------------------------------------
            // Parameter / Profiling / Availability
            // ----------------------------------------------------------------
            auto max_parameter_size     = dev.get_info<sycl::info::device::max_parameter_size>();
            auto profiling_timer_res    = dev.get_info<sycl::info::device::profiling_timer_resolution>();
            auto is_available           = dev.get_info<sycl::info::device::is_available>();
            auto is_compiler_available  = dev.get_info<sycl::info::device::is_compiler_available>();
            auto is_linker_available    = dev.get_info<sycl::info::device::is_linker_available>();
            auto reference_count        = dev.get_info<sycl::info::device::reference_count>();
            printf("  max_parameter_size (max size of kernel arguments in bytes): %zu\n", max_parameter_size);
            printf("  profiling_timer_resolution (profiling timer resolution in nanoseconds): %zu\n", profiling_timer_res);
            try {
                auto printf_buffer_size = dev.get_info<sycl::info::device::printf_buffer_size>();
                printf("  printf_buffer_size (max printf buffer size in bytes, deprecated): %zu\n", printf_buffer_size);
            } catch (...) { printf("  printf_buffer_size (max printf buffer size in bytes, deprecated): n/a\n"); }
            try {
                auto preferred_interop_sync = dev.get_info<sycl::info::device::preferred_interop_user_sync>();
                printf("  preferred_interop_user_sync (user responsible for sync when sharing with graphics API, deprecated): %d\n", (int)preferred_interop_sync);
            } catch (...) { printf("  preferred_interop_user_sync (deprecated): n/a\n"); }
            printf("  is_available (device is available): %d\n", (int)is_available);
            printf("  is_compiler_available (online compiler available, deprecated): %d\n", (int)is_compiler_available);
            printf("  is_linker_available (online linker available, deprecated): %d\n", (int)is_linker_available);
            printf("  reference_count (reference count of the device object): %u\n", reference_count);

            // ----------------------------------------------------------------
            // Built-in Kernels
            // ----------------------------------------------------------------
            auto builtin_kernel_ids = dev.get_info<sycl::info::device::built_in_kernel_ids>();
            printf("  built_in_kernel_ids (number of built-in kernels): %zu\n", builtin_kernel_ids.size());

            // ----------------------------------------------------------------
            // Partitioning
            // ----------------------------------------------------------------
            auto max_sub_devices  = dev.get_info<sycl::info::device::partition_max_sub_devices>();
            printf("  partition_max_sub_devices (max sub-devices when partitioned): %u\n", max_sub_devices);

            auto partition_props = dev.get_info<sycl::info::device::partition_properties>();
            printf("  partition_properties (supported partition schemes):");
            if (partition_props.empty()) printf(" none");
            for (auto pp : partition_props) {
                if      (pp == sycl::info::partition_property::partition_equally)            printf(" partition_equally");
                else if (pp == sycl::info::partition_property::partition_by_counts)          printf(" partition_by_counts");
                else if (pp == sycl::info::partition_property::partition_by_affinity_domain) printf(" partition_by_affinity_domain");
                else                                                                         printf(" unknown");
            }
            printf("\n");

            auto affinity_domains = dev.get_info<sycl::info::device::partition_affinity_domains>();
            printf("  partition_affinity_domains (supported affinity domains for partition_by_affinity_domain):");
            if (affinity_domains.empty()) printf(" none");
            for (auto ad : affinity_domains) {
                if      (ad == sycl::info::partition_affinity_domain::numa)               printf(" numa");
                else if (ad == sycl::info::partition_affinity_domain::L4_cache)           printf(" L4_cache");
                else if (ad == sycl::info::partition_affinity_domain::L3_cache)           printf(" L3_cache");
                else if (ad == sycl::info::partition_affinity_domain::L2_cache)           printf(" L2_cache");
                else if (ad == sycl::info::partition_affinity_domain::L1_cache)           printf(" L1_cache");
                else if (ad == sycl::info::partition_affinity_domain::next_partitionable) printf(" next_partitionable");
                else                                                                      printf(" unknown");
            }
            printf("\n");

            auto part_type_prop = dev.get_info<sycl::info::device::partition_type_property>();
            const char* part_type_str = "no_partition";
            if      (part_type_prop == sycl::info::partition_property::partition_equally)            part_type_str = "partition_equally";
            else if (part_type_prop == sycl::info::partition_property::partition_by_counts)          part_type_str = "partition_by_counts";
            else if (part_type_prop == sycl::info::partition_property::partition_by_affinity_domain) part_type_str = "partition_by_affinity_domain";
            printf("  partition_type_property (partition scheme used to create this sub-device): %s\n", part_type_str);

            auto part_type_affinity = dev.get_info<sycl::info::device::partition_type_affinity_domain>();
            const char* part_affinity_str = "not_applicable";
            if      (part_type_affinity == sycl::info::partition_affinity_domain::numa)               part_affinity_str = "numa";
            else if (part_type_affinity == sycl::info::partition_affinity_domain::L4_cache)           part_affinity_str = "L4_cache";
            else if (part_type_affinity == sycl::info::partition_affinity_domain::L3_cache)           part_affinity_str = "L3_cache";
            else if (part_type_affinity == sycl::info::partition_affinity_domain::L2_cache)           part_affinity_str = "L2_cache";
            else if (part_type_affinity == sycl::info::partition_affinity_domain::L1_cache)           part_affinity_str = "L1_cache";
            else if (part_type_affinity == sycl::info::partition_affinity_domain::next_partitionable) part_affinity_str = "next_partitionable";
            printf("  partition_type_affinity_domain (affinity domain used to create this sub-device): %s\n", part_affinity_str);

            // ----------------------------------------------------------------
            // Atomics
            // ----------------------------------------------------------------
            auto print_memory_orders = [](const char* label, const char* desc, auto caps) {
                printf("  %s (%s):", label, desc);
                for (auto cap : caps) {
                    if      (cap == sycl::memory_order::relaxed) printf(" relaxed");
                    else if (cap == sycl::memory_order::acquire) printf(" acquire");
                    else if (cap == sycl::memory_order::release) printf(" release");
                    else if (cap == sycl::memory_order::acq_rel) printf(" acq_rel");
                    else if (cap == sycl::memory_order::seq_cst) printf(" seq_cst");
                }
                printf("\n");
            };
            auto print_memory_scopes = [](const char* label, const char* desc, auto caps) {
                printf("  %s (%s):", label, desc);
                for (auto cap : caps) {
                    if      (cap == sycl::memory_scope::work_item)  printf(" work_item");
                    else if (cap == sycl::memory_scope::sub_group)  printf(" sub_group");
                    else if (cap == sycl::memory_scope::work_group) printf(" work_group");
                    else if (cap == sycl::memory_scope::device)     printf(" device");
                    else if (cap == sycl::memory_scope::system)     printf(" system");
                }
                printf("\n");
            };
            print_memory_orders("atomic_memory_order_capabilities", "supported memory orders for atomic operations",       dev.get_info<sycl::info::device::atomic_memory_order_capabilities>());
            print_memory_orders("atomic_fence_order_capabilities",  "supported memory orders for atomic fence operations",  dev.get_info<sycl::info::device::atomic_fence_order_capabilities>());
            print_memory_scopes("atomic_memory_scope_capabilities", "supported memory scopes for atomic operations",        dev.get_info<sycl::info::device::atomic_memory_scope_capabilities>());
            print_memory_scopes("atomic_fence_scope_capabilities",  "supported memory scopes for atomic fence operations",  dev.get_info<sycl::info::device::atomic_fence_scope_capabilities>());

            // ----------------------------------------------------------------
            // Preferred / Native Vector Widths
            // ----------------------------------------------------------------
            printf("  preferred_vector_width_char (preferred native vector width for char): %u\n",    dev.get_info<sycl::info::device::preferred_vector_width_char>());
            printf("  preferred_vector_width_short (preferred native vector width for short): %u\n",  dev.get_info<sycl::info::device::preferred_vector_width_short>());
            printf("  preferred_vector_width_int (preferred native vector width for int): %u\n",      dev.get_info<sycl::info::device::preferred_vector_width_int>());
            printf("  preferred_vector_width_long (preferred native vector width for long): %u\n",    dev.get_info<sycl::info::device::preferred_vector_width_long>());
            printf("  preferred_vector_width_float (preferred native vector width for float): %u\n",  dev.get_info<sycl::info::device::preferred_vector_width_float>());
            printf("  preferred_vector_width_double (preferred native vector width for double): %u\n",dev.get_info<sycl::info::device::preferred_vector_width_double>());
            printf("  preferred_vector_width_half (preferred native vector width for half): %u\n",    dev.get_info<sycl::info::device::preferred_vector_width_half>());
            printf("  native_vector_width_char (native ISA vector width for char): %u\n",             dev.get_info<sycl::info::device::native_vector_width_char>());
            printf("  native_vector_width_short (native ISA vector width for short): %u\n",           dev.get_info<sycl::info::device::native_vector_width_short>());
            printf("  native_vector_width_int (native ISA vector width for int): %u\n",               dev.get_info<sycl::info::device::native_vector_width_int>());
            printf("  native_vector_width_long (native ISA vector width for long): %u\n",             dev.get_info<sycl::info::device::native_vector_width_long>());
            printf("  native_vector_width_float (native ISA vector width for float): %u\n",           dev.get_info<sycl::info::device::native_vector_width_float>());
            printf("  native_vector_width_double (native ISA vector width for double): %u\n",         dev.get_info<sycl::info::device::native_vector_width_double>());
            printf("  native_vector_width_half (native ISA vector width for half): %u\n",             dev.get_info<sycl::info::device::native_vector_width_half>());

            // ----------------------------------------------------------------
            // FP Capabilities
            // ----------------------------------------------------------------
            auto print_fp_config = [](const char* label, const char* desc, auto caps) {
                printf("  %s (%s):", label, desc);
                for (auto cap : caps) {
                    if      (cap == sycl::info::fp_config::denorm)                        printf(" denorm");
                    else if (cap == sycl::info::fp_config::inf_nan)                       printf(" inf_nan");
                    else if (cap == sycl::info::fp_config::round_to_nearest)              printf(" round_to_nearest");
                    else if (cap == sycl::info::fp_config::round_to_zero)                 printf(" round_to_zero");
                    else if (cap == sycl::info::fp_config::round_to_inf)                  printf(" round_to_inf");
                    else if (cap == sycl::info::fp_config::fma)                           printf(" fma");
                    else if (cap == sycl::info::fp_config::correctly_rounded_divide_sqrt) printf(" correctly_rounded_divide_sqrt");
                    else if (cap == sycl::info::fp_config::soft_float)                    printf(" soft_float");
                }
                printf("\n");
            };
            print_fp_config("half_fp_config",   "half precision floating-point capabilities",   dev.get_info<sycl::info::device::half_fp_config>());
            print_fp_config("single_fp_config", "single precision floating-point capabilities", dev.get_info<sycl::info::device::single_fp_config>());
            print_fp_config("double_fp_config", "double precision floating-point capabilities", dev.get_info<sycl::info::device::double_fp_config>());

            // ----------------------------------------------------------------
            // Error Correction / Memory Coherency / Endianness
            // ----------------------------------------------------------------
            printf("  error_correction_support (device supports ECC): %d\n",
                   (int)dev.get_info<sycl::info::device::error_correction_support>());
            printf("  host_unified_memory (device and host share unified memory, deprecated): %d\n",
                   (int)dev.get_info<sycl::info::device::host_unified_memory>());
            printf("  is_endian_little (device is little endian, deprecated): %d\n",
                   (int)dev.get_info<sycl::info::device::is_endian_little>());

            // ----------------------------------------------------------------
            // Standard SYCL 2020 Aspects
            // ----------------------------------------------------------------
            printf("  -- SYCL 2020 Aspects --\n");
            printf("  aspect::cpu (device is a CPU): %d\n",                              (int)dev.has(sycl::aspect::cpu));
            printf("  aspect::gpu (device is a GPU): %d\n",                              (int)dev.has(sycl::aspect::gpu));
            printf("  aspect::accelerator (device is an accelerator): %d\n",             (int)dev.has(sycl::aspect::accelerator));
            printf("  aspect::custom (device is a custom device): %d\n",                 (int)dev.has(sycl::aspect::custom));
            printf("  aspect::emulated (device is emulated): %d\n",                      (int)dev.has(sycl::aspect::emulated));
            printf("  aspect::host_debuggable (device supports host debugging): %d\n",   (int)dev.has(sycl::aspect::host_debuggable));
            printf("  aspect::fp16 (device supports half precision): %d\n",              (int)dev.has(sycl::aspect::fp16));
            printf("  aspect::fp64 (device supports double precision): %d\n",            (int)dev.has(sycl::aspect::fp64));
            printf("  aspect::atomic64 (device supports 64-bit atomics): %d\n",          (int)dev.has(sycl::aspect::atomic64));
            printf("  aspect::image (device supports SYCL 2020 images): %d\n",           (int)dev.has(sycl::aspect::image));
            printf("  aspect::online_compiler (device has online compiler): %d\n",       (int)dev.has(sycl::aspect::online_compiler));
            printf("  aspect::online_linker (device has online linker): %d\n",           (int)dev.has(sycl::aspect::online_linker));
            printf("  aspect::queue_profiling (device supports queue profiling): %d\n",  (int)dev.has(sycl::aspect::queue_profiling));
            printf("  aspect::usm_device_allocations (USM device allocations): %d\n",    (int)dev.has(sycl::aspect::usm_device_allocations));
            printf("  aspect::usm_host_allocations (USM host allocations): %d\n",        (int)dev.has(sycl::aspect::usm_host_allocations));
            printf("  aspect::usm_shared_allocations (USM shared allocations): %d\n",    (int)dev.has(sycl::aspect::usm_shared_allocations));
            printf("  aspect::usm_system_allocations (USM system allocations): %d\n",    (int)dev.has(sycl::aspect::usm_system_allocations));
            printf("  aspect::usm_atomic_host_allocations (USM atomic host allocations): %d\n",    (int)dev.has(sycl::aspect::usm_atomic_host_allocations));
            printf("  aspect::usm_atomic_shared_allocations (USM atomic shared allocations): %d\n",(int)dev.has(sycl::aspect::usm_atomic_shared_allocations));

            // ----------------------------------------------------------------
            // Intel-specific Aspects
            // ----------------------------------------------------------------
            printf("  -- Intel-specific Aspects --\n");
            printf("  aspect::ext_intel_legacy_image (device supports SYCL 1.2.1 images): %d\n",          (int)dev.has(sycl::aspect::ext_intel_legacy_image));
            printf("  aspect::ext_intel_esimd (device supports ESIMD extension): %d\n",                   (int)dev.has(sycl::aspect::ext_intel_esimd));
            printf("  aspect::ext_intel_free_memory (free_memory query available): %d\n",                 (int)dev.has(sycl::aspect::ext_intel_free_memory));
            printf("  aspect::ext_intel_device_id (device_id query available): %d\n",                     (int)dev.has(sycl::aspect::ext_intel_device_id));
            printf("  aspect::ext_intel_memory_clock_rate (memory_clock_rate query available): %d\n",     (int)dev.has(sycl::aspect::ext_intel_memory_clock_rate));
            printf("  aspect::ext_intel_memory_bus_width (memory_bus_width query available): %d\n",       (int)dev.has(sycl::aspect::ext_intel_memory_bus_width));
            printf("  aspect::ext_intel_device_info_uuid (uuid query available): %d\n",                   (int)dev.has(sycl::aspect::ext_intel_device_info_uuid));
            printf("  aspect::ext_intel_pci_address (pci_address query available): %d\n",                 (int)dev.has(sycl::aspect::ext_intel_pci_address));
            printf("  aspect::ext_intel_gpu_eu_count (gpu_eu_count query available): %d\n",               (int)dev.has(sycl::aspect::ext_intel_gpu_eu_count));
            printf("  aspect::ext_intel_gpu_eu_simd_width (gpu_eu_simd_width query available): %d\n",     (int)dev.has(sycl::aspect::ext_intel_gpu_eu_simd_width));
            printf("  aspect::ext_intel_gpu_slices (gpu_slices query available): %d\n",                   (int)dev.has(sycl::aspect::ext_intel_gpu_slices));
            printf("  aspect::ext_intel_gpu_subslices_per_slice (gpu_subslices_per_slice query available): %d\n", (int)dev.has(sycl::aspect::ext_intel_gpu_subslices_per_slice));
            printf("  aspect::ext_intel_gpu_eu_count_per_subslice (gpu_eu_count_per_subslice query available): %d\n", (int)dev.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice));
            printf("  aspect::ext_intel_gpu_hw_threads_per_eu (gpu_hw_threads_per_eu query available): %d\n", (int)dev.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu));
            printf("  aspect::ext_intel_max_mem_bandwidth (max_mem_bandwidth query available): %d\n",     (int)dev.has(sycl::aspect::ext_intel_max_mem_bandwidth));
            printf("  aspect::ext_intel_device_info_luid (luid query available): %d\n",                   (int)dev.has(sycl::aspect::ext_intel_device_info_luid));
            printf("  aspect::ext_intel_device_info_node_mask (node_mask query available): %d\n",         (int)dev.has(sycl::aspect::ext_intel_device_info_node_mask));
            printf("  aspect::ext_intel_fan_speed (fan_speed query available): %d\n",                     (int)dev.has(sycl::aspect::ext_intel_fan_speed));
            printf("  aspect::ext_intel_power_limits (power limits query available): %d\n",               (int)dev.has(sycl::aspect::ext_intel_power_limits));
            printf("  aspect::ext_oneapi_srgb (device supports sRGB writes): %d\n",                      (int)dev.has(sycl::aspect::ext_oneapi_srgb));

            // ----------------------------------------------------------------
            // Intel Extension Properties (ext::intel::info::device::*)
            // ----------------------------------------------------------------
            printf("  -- Intel Extension Properties --\n");

            if (dev.has(sycl::aspect::ext_intel_device_id)) {
                printf("  ext_intel_device_id (PCI device ID): 0x%x\n",
                       dev.get_info<sycl::ext::intel::info::device::device_id>());
            }
            if (dev.has(sycl::aspect::ext_intel_pci_address)) {
                auto pci_addr = dev.get_info<sycl::ext::intel::info::device::pci_address>();
                printf("  ext_intel_pci_address (PCI bus:device.function address): %s\n", pci_addr.c_str());
            }
            if (dev.has(sycl::aspect::ext_intel_device_info_uuid)) {
                auto uuid = dev.get_info<sycl::ext::intel::info::device::uuid>();
                printf("  ext_intel_uuid (device universally unique identifier):");
                for (int i = 0; i < (int)uuid.size(); i++) {
                    if (i == 4 || i == 6 || i == 8 || i == 10) printf("-");
                    printf("%02x", (unsigned)uuid[i]);
                }
                printf("\n");
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_eu_count)) {
                printf("  ext_intel_gpu_eu_count (total number of Execution Units): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
                printf("  ext_intel_gpu_eu_simd_width (physical SIMD width of an EU): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_slices)) {
                printf("  ext_intel_gpu_slices (number of slices in the GPU): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_slices>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
                printf("  ext_intel_gpu_subslices_per_slice (number of subslices per slice): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_eu_count_per_subslice)) {
                printf("  ext_intel_gpu_eu_count_per_subslice (number of EUs per subslice): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_eu_count_per_subslice>());
            }
            if (dev.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
                printf("  ext_intel_gpu_hw_threads_per_eu (number of hardware threads per EU): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::gpu_hw_threads_per_eu>());
            }
            if (dev.has(sycl::aspect::ext_intel_max_mem_bandwidth)) {
                printf("  ext_intel_max_mem_bandwidth (max memory bandwidth in bytes/second): %llu\n",
                       (unsigned long long)dev.get_info<sycl::ext::intel::info::device::max_mem_bandwidth>());
            }
            if (dev.has(sycl::aspect::ext_intel_free_memory)) {
                printf("  ext_intel_free_memory (free global memory in bytes): %llu\n",
                       (unsigned long long)dev.get_info<sycl::ext::intel::info::device::free_memory>());
            }
            if (dev.has(sycl::aspect::ext_intel_memory_clock_rate)) {
                printf("  ext_intel_memory_clock_rate (memory clock rate in kHz): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>());
            }
            if (dev.has(sycl::aspect::ext_intel_memory_bus_width)) {
                printf("  ext_intel_memory_bus_width (memory bus width in bits): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
            }
            try {
                auto idx = dev.get_info<sycl::ext::intel::info::device::max_compute_queue_indices>();
                printf("  ext_intel_max_compute_queue_indices (max queue indices for ordered queues): %d\n", idx);
            } catch (...) {}

            if (dev.has(sycl::aspect::ext_intel_device_info_luid)) {
                auto luid = dev.get_info<sycl::ext::intel::info::device::luid>();
                printf("  ext_intel_luid (locally unique identifier):");
                for (auto b : luid) printf(" %02x", (unsigned)b);
                printf("\n");
                printf("  ext_intel_node_mask (node mask for the device): %u\n",
                       dev.get_info<sycl::ext::intel::info::device::node_mask>());
            }
            if (dev.has(sycl::aspect::ext_intel_fan_speed)) {
                printf("  ext_intel_fan_speed (current fan speed in RPM, -1 if not spinning): %d\n",
                       dev.get_info<sycl::ext::intel::info::device::fan_speed>());
            }
            if (dev.has(sycl::aspect::ext_intel_power_limits)) {
                printf("  ext_intel_min_power_limit (minimum power limit in milliwatts): %d\n",
                       dev.get_info<sycl::ext::intel::info::device::min_power_limit>());
                printf("  ext_intel_max_power_limit (maximum power limit in milliwatts): %d\n",
                       dev.get_info<sycl::ext::intel::info::device::max_power_limit>());
            }

            // ----------------------------------------------------------------
            // ext::oneapi::experimental properties
            // ----------------------------------------------------------------
            printf("  -- oneAPI Experimental Properties --\n");
            try {
                auto max_wg = dev.get_info<sycl::ext::oneapi::experimental::info::device::max_global_work_groups>();
                printf("  ext_oneapi_max_global_work_groups (max total number of work-groups): %zu\n", max_wg);
            } catch (...) {}
            try {
                auto max_wg3 = dev.get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
                printf("  ext_oneapi_max_work_groups_3d (max work-groups per dimension): [%zu, %zu, %zu]\n",
                       max_wg3[0], max_wg3[1], max_wg3[2]);
            } catch (...) {}

            // ----------------------------------------------------------------
            // Derived Metrics (matching CUDA's derived section)
            // ----------------------------------------------------------------
            printf("  -- Derived Metrics --\n");

            // Peak Memory Bandwidth
            if (dev.has(sycl::aspect::ext_intel_max_mem_bandwidth)) {
                auto max_bw = dev.get_info<sycl::ext::intel::info::device::max_mem_bandwidth>();
                printf("  Peak Memory Bandwidth (GB/s): %.2f\n", (double)max_bw / 1.0e9);
            } else if (dev.has(sycl::aspect::ext_intel_memory_clock_rate) &&
                       dev.has(sycl::aspect::ext_intel_memory_bus_width)) {
                auto mem_clock_khz = dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
                auto mem_bus_bits  = dev.get_info<sycl::ext::intel::info::device::memory_bus_width>();
                // Same formula as CUDA: 2.0 * clockRate_kHz * (busWidth_bits / 8) / 1e6
                double peak_bw_gbs = 2.0 * mem_clock_khz * (mem_bus_bits / 8) / 1.0e6;
                printf("  Peak Memory Bandwidth (GB/s): %.2f\n", peak_bw_gbs);
            } else {
                printf("  Peak Memory Bandwidth (GB/s): n/a (memory clock/bus width not available)\n");
            }

            // Peak Compute Throughput (FP32)
            if (dev.has(sycl::aspect::ext_intel_gpu_eu_count) &&
                dev.has(sycl::aspect::ext_intel_gpu_eu_simd_width)) {
                auto eu_count    = dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
                auto simd_width  = dev.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();
                // Each EU can issue 2 FP32 ops per cycle (FMA = multiply + add)
                // Peak FP32 GFLOPS = EU_count * SIMD_width * 2 (FMA) * clock_MHz / 1000
                double peak_gflops = (double)eu_count * simd_width * 2.0 * max_clock_frequency / 1000.0;
                printf("  Peak FP32 Throughput (GFLOPS, estimated): %.2f\n", peak_gflops);
                printf("    (formula: EU_count=%u * SIMD_width=%u * 2_FMA * clock_MHz=%u / 1000)\n",
                       eu_count, simd_width, max_clock_frequency);
            } else {
                printf("  Peak FP32 Throughput (GFLOPS): n/a (EU count/SIMD width not available)\n");
            }

            // Global Memory Size (human-readable)
            printf("  Global Memory (GB): %.2f\n", (double)global_mem_size / (1024.0 * 1024.0 * 1024.0));
            printf("  Local Memory (KB): %.2f\n",  (double)local_mem_size / 1024.0);

            printf("\n");
            device_idx++;
        }
    }

    if (device_idx == 0) {
        printf("No XPU/GPU devices found via SYCL.\n");
    } else {
        printf("Total XPU devices: %d\n", device_idx);
    }
}

#pragma clang diagnostic pop
