/*
 * glibc < 2.38 compatibility shim for the C23 strtol* family.
 *
 * Why this file exists
 * --------------------
 *
 * glibc 2.38 (Aug 2023) added `__isoc23_strtol`, `__isoc23_strtoll`,
 * `__isoc23_strtoul`, and `__isoc23_strtoull` as canonical C23
 * implementations of strtol*. When you compile C/C++ code with a
 * recent toolchain (gcc >= 13) and the headers see C23/C++23 mode
 * (or `_GNU_SOURCE`), `<stdlib.h>` redirects every call to
 * `strtoll(...)` to `__isoc23_strtoll(...)` via a transparent inline.
 *
 * The ONNX Runtime prebuilt artifacts that we statically link
 * (downloaded by `ort-download-binaries-rustls-tls` via fastembed)
 * are compiled with gcc-14.2.1 on a glibc >= 2.38 toolchain. They
 * therefore reference `__isoc23_*` symbols. Our wheel build runs
 * in `manylinux_2_28` (glibc 2.28 baseline) but the linker doesn't
 * complain because these are deferred (DT_NEEDED-style) symbols
 * resolved at runtime — and the manylinux_2_28 host's glibc has
 * them, so the link-time check passes.
 *
 * On the END USER's runtime, however, glibc < 2.38 has none of
 * these symbols, and `import headroom._core` fails with:
 *
 *     ImportError: undefined symbol: __isoc23_strtoll
 *
 * (Reported in issue #355; first hit by users on Ubuntu 22.04 +
 * Conda Python 3.12 environments where libc.so.6 is glibc 2.35.)
 *
 * The fix
 * -------
 *
 * Define the four `__isoc23_*` symbols as weak aliases for the
 * older `strtoll` family. The dynamic linker resolves symbols in
 * library load order: when glibc has the strong symbol (>=2.38)
 * its definition wins; when it doesn't (<2.38) our weak symbol
 * is used. Either way, `_core.so` loads.
 *
 * The C23 vs pre-C23 behavioural difference is binary-literal
 * support: `__isoc23_strtoll` accepts strings like "0b1010"
 * whereas `strtoll` returns 0 for those. Our static-link callsites
 * (deep inside ORT's protobuf parsers) only pass decimal /
 * hexadecimal numerics, so the fallback is functionally identical.
 *
 * References:
 * - https://sourceware.org/glibc/wiki/Release/2.38
 * - https://github.com/pypa/manylinux/issues/1725
 * - issue #355: tests/test_rust_core_smoke.py was the canary that
 *   surfaced this on user installs.
 *
 * This file is compiled and linked into `_core.so` only on Linux
 * (gated in build.rs). macOS and Windows have neither glibc nor
 * this symbol family.
 */

#include <stdlib.h>

#define ALIAS_TO(target) \
    __attribute__((weak, alias(#target)))

long __isoc23_strtol(const char *nptr, char **endptr, int base) ALIAS_TO(strtol);

long long __isoc23_strtoll(const char *nptr, char **endptr, int base) ALIAS_TO(strtoll);

unsigned long __isoc23_strtoul(const char *nptr, char **endptr, int base) ALIAS_TO(strtoul);

unsigned long long __isoc23_strtoull(const char *nptr, char **endptr, int base) ALIAS_TO(strtoull);
