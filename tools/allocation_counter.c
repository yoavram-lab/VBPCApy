/* Linux/glibc diagnostic only; never linked into the installed extensions. */
#include <stddef.h>
#include <stdatomic.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);

static _Atomic unsigned long long allocation_count = 0;
static _Atomic int enabled = 0;

void vbpca_allocation_start(void) {
    atomic_store(&allocation_count, 0);
    atomic_store(&enabled, 1);
}

unsigned long long vbpca_allocation_stop(void) {
    atomic_store(&enabled, 0);
    return atomic_load(&allocation_count);
}

void *malloc(size_t bytes) {
    if (atomic_load(&enabled)) atomic_fetch_add(&allocation_count, 1);
    return __libc_malloc(bytes);
}

void *calloc(size_t count, size_t bytes) {
    if (atomic_load(&enabled)) atomic_fetch_add(&allocation_count, 1);
    return __libc_calloc(count, bytes);
}

void *realloc(void *pointer, size_t bytes) {
    if (atomic_load(&enabled)) atomic_fetch_add(&allocation_count, 1);
    return __libc_realloc(pointer, bytes);
}
