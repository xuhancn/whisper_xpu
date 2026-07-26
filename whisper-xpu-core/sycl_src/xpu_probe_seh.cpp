// SEH wrapper — xpuDeviceProp.cpp (unmodified) crashes on some machines
// when sycl8.dll has a null fn ptr during platform enumeration.
// __except(1) catches the AV; the function prints nothing on crash.
// icpx supports __try with C++ objects.

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" void print_all_device_properties();

extern "C" EXPORT void print_all_device_properties_seh() {
    __try {
        print_all_device_properties();
    } __except(1) {
        // sycl8.dll platform enumeration crashed
    }
}
