// Compile a tiny C shim that provides weak aliases for the C23
// strtol family (`__isoc23_strtol`, `__isoc23_strtoll`, etc.).
//
// glibc < 2.38 doesn't ship these symbols. Static dependencies in
// `_core.so` (notably the prebuilt ONNX Runtime artifacts compiled
// with gcc 14.x) reference them, so without this shim the wheel
// fails to import on Ubuntu 22.04, Conda envs with libc 2.35, etc.
//
// See `glibc_compat.c` for the full background. The shim is
// Linux-only — macOS and Windows don't ship glibc.
//
// Issue: #355 (https://github.com/chopratejas/headroom/issues/355)

fn main() {
    println!("cargo:rerun-if-changed=glibc_compat.c");
    println!("cargo:rerun-if-changed=build.rs");

    // The shim is glibc-specific. Skip on every other target: macOS
    // uses Darwin libc, Windows has MSVCRT, musl handles strtoll
    // identically and never emits __isoc23_*.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    if target_os != "linux" || target_env != "gnu" {
        return;
    }

    cc::Build::new()
        .file("glibc_compat.c")
        // -fPIC because we link into a cdylib. -O2 for size — the
        // file is ~10 lines but every byte counts in a wheel that's
        // already 35 MiB.
        .flag_if_supported("-fPIC")
        .opt_level(2)
        // Suppress GCC's "weak attribute, alias to symbol that may
        // not exist" warning. `strtoll` etc. always exist; the
        // weak attribute is for OUR `__isoc23_*` definitions, not
        // for the alias targets.
        .flag_if_supported("-Wno-attribute-alias")
        .compile("headroom_glibc_compat");
}
