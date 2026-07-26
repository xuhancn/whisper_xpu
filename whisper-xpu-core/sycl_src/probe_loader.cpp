// Loader: sets ZES_ENABLE_SYSMAN, then loads xpu_device_prop.dll
// and calls print_all_device_properties().
#include <windows.h>
#include <cstdio>
#include <cstdlib>

int main() {
    fprintf(stdout, "[loader] setting ZES_ENABLE_SYSMAN=1\n");
    fflush(stdout);
    SetEnvironmentVariableA("ZES_ENABLE_SYSMAN", "1");

    // Mimic PyTorch: warm up sycl8.dll by loading it before any SYCL call.
    // PyTorch's c10_xpu.dll depends on sycl8.dll, so the Windows loader
    // resolves sycl8.dll before PyTorch calls get_platforms().
    fprintf(stdout, "[loader] pre-loading sycl8.dll\n");
    fflush(stdout);
    HMODULE hSycl = LoadLibraryA("sycl8.dll");
    if (hSycl) {
        fprintf(stdout, "[loader] sycl8.dll loaded OK\n");
    } else {
        fprintf(stdout, "[loader] sycl8.dll not available, continuing anyway\n");
    }
    fflush(stdout);

    fprintf(stdout, "[loader] loading xpu_device_prop.dll\n");
    fflush(stdout);
    HMODULE h = LoadLibraryA("xpu_device_prop.dll");
    if (!h) {
        fprintf(stderr, "[loader] LoadLibrary failed: %lu\n", GetLastError());
        return 1;
    }
    fprintf(stdout, "[loader] DLL loaded OK\n");
    fflush(stdout);

    using Fn = void (*)();
    auto fn = (Fn)GetProcAddress(h, "print_all_device_properties");
    if (!fn) {
        fprintf(stderr, "[loader] GetProcAddress failed\n");
        FreeLibrary(h);
        return 1;
    }
    fprintf(stdout, "[loader] calling print_all_device_properties...\n");
    fflush(stdout);

    fn();

    fprintf(stdout, "[loader] done\n");
    if (hSycl) FreeLibrary(hSycl);
    FreeLibrary(h);
    return 0;
}
