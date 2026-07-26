// Loader: sets ZES_ENABLE_SYSMAN, then loads xpu_device_prop.dll
// and calls print_all_device_properties().
#include <windows.h>
#include <cstdio>
#include <cstdlib>

int main() {
    fprintf(stdout, "[loader] setting ZES_ENABLE_SYSMAN=1\n");
    fflush(stdout);
    SetEnvironmentVariableA("ZES_ENABLE_SYSMAN", "1");

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
    auto fn = (Fn)GetProcAddress(h, "print_all_device_properties_seh");
    if (!fn) {
        fprintf(stderr, "[loader] GetProcAddress failed\n");
        FreeLibrary(h);
        return 1;
    }
    fprintf(stdout, "[loader] calling print_all_device_properties...\n");
    fflush(stdout);

    fn();

    fprintf(stdout, "[loader] done\n");
    FreeLibrary(h);
    return 0;
}
