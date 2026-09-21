// Minimal check macro shared by the CTest executables: counts failures, prints file:line and
// a printf-style message, never aborts (so one run reports every failing expectation).
#pragma once

#include <cstdio>

namespace edgevision_test {
inline int failures = 0;
}

#define EXPECT(cond, ...)                                    \
    do {                                                     \
        if (!(cond)) {                                       \
            ++edgevision_test::failures;                     \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
        }                                                    \
    } while (0)

// Runs `expr` and checks that it throws std::runtime_error whose message contains `needle`.
#define EXPECT_THROWS(expr, needle)                                                                              \
    do {                                                                                                         \
        bool thrown = false;                                                                                     \
        try {                                                                                                    \
            (void)(expr);                                                                                        \
        } catch (const std::runtime_error& e) {                                                                  \
            thrown = true;                                                                                       \
            EXPECT(std::string(e.what()).find(needle) != std::string::npos, "message '%s' lacks '%s'", e.what(), \
                   needle);                                                                                      \
        }                                                                                                        \
        EXPECT(thrown, "expected an exception containing '%s'", needle);                                         \
    } while (0)

// Return value for main(): 0 when every EXPECT passed.
inline int test_result(const char* suite) {
    if (edgevision_test::failures == 0) std::printf("%s tests: OK\n", suite);
    return edgevision_test::failures == 0 ? 0 : 1;
}
