#ifndef DS4_METAL_SOURCE_H
#define DS4_METAL_SOURCE_H

#import <Foundation/Foundation.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Startup-only helpers. No Metal device, command queue, or model is needed. */
typedef enum {
    DS4_METAL_SOURCE_MISSING,
    DS4_METAL_SOURCE_OVERRIDE,
    DS4_METAL_SOURCE_EXECUTABLE,
    DS4_METAL_SOURCE_CWD,
} ds4_metal_source_origin;

static inline NSString *ds4_metal_executable_path(void) {
    uint32_t length = 0;
    (void)_NSGetExecutablePath(NULL, &length);
    if (!length) return nil;
    char *buffer = malloc(length);
    if (!buffer) return nil;
    NSString *path = nil;
    if (_NSGetExecutablePath(buffer, &length) == 0)
        path = [[NSFileManager defaultManager]
            stringWithFileSystemRepresentation:buffer length:strlen(buffer)];
    free(buffer);
    if (path && ![path isAbsolutePath])
        path = [[[NSFileManager defaultManager] currentDirectoryPath]
            stringByAppendingPathComponent:path];
    return [[path stringByStandardizingPath] stringByResolvingSymlinksInPath];
}

static inline NSString *ds4_metal_source_path(
        NSString *executable, NSString *relative, const char *override_path,
        ds4_metal_source_origin *origin) {
    NSFileManager *fm = [NSFileManager defaultManager];
    if (origin) *origin = DS4_METAL_SOURCE_MISSING;
    if (override_path && override_path[0]) {
        NSString *path = [NSString stringWithUTF8String:override_path];
        if (path && [fm fileExistsAtPath:path]) {
            if (origin) *origin = DS4_METAL_SOURCE_OVERRIDE;
            return path;
        }
    }
    if (executable) {
        NSString *path = [[executable stringByDeletingLastPathComponent]
            stringByAppendingPathComponent:relative];
        if ([fm fileExistsAtPath:path]) {
            if (origin) *origin = DS4_METAL_SOURCE_EXECUTABLE;
            return path;
        }
    }
    if ([fm fileExistsAtPath:relative]) {
        if (origin) *origin = DS4_METAL_SOURCE_CWD;
        return relative;
    }
    return nil;
}

static inline void ds4_metal_digest_bytes(CC_SHA256_CTX *ctx,
                                         const void *bytes, size_t length) {
    const unsigned char *p = bytes;
    while (length) {
        const CC_LONG count = length > UINT32_MAX ? UINT32_MAX : (CC_LONG)length;
        CC_SHA256_Update(ctx, p, count);
        p += count;
        length -= count;
    }
}

static inline void ds4_metal_digest_u64le(CC_SHA256_CTX *ctx, uint64_t value) {
    unsigned char bytes[8];
    for (unsigned i = 0; i < 8; i++) bytes[i] = (unsigned char)(value >> (8u * i));
    ds4_metal_digest_bytes(ctx, bytes, sizeof(bytes));
}

/* Reproducible framing: domain, then ordered parts of u64LE name length,
 * UTF-8 logical name, u64LE data length, and raw data. Paths used to find a
 * part never enter this digest; embedded source is the first named part. */
static inline void ds4_metal_digest_init(CC_SHA256_CTX *ctx) {
    static const char domain[] = "DS4-METAL-SOURCE-v1\n";
    CC_SHA256_Init(ctx);
    ds4_metal_digest_bytes(ctx, domain, sizeof(domain) - 1u);
}

static inline void ds4_metal_digest_part(CC_SHA256_CTX *ctx,
                                        NSString *name, NSData *data) {
    NSData *name_bytes = [name dataUsingEncoding:NSUTF8StringEncoding];
    ds4_metal_digest_u64le(ctx, [name_bytes length]);
    ds4_metal_digest_bytes(ctx, [name_bytes bytes], [name_bytes length]);
    ds4_metal_digest_u64le(ctx, [data length]);
    ds4_metal_digest_bytes(ctx, [data bytes], [data length]);
}

static inline NSString *ds4_metal_digest_finish(CC_SHA256_CTX *ctx) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    char hex[CC_SHA256_DIGEST_LENGTH * 2u + 1u];
    CC_SHA256_Final(digest, ctx);
    for (unsigned i = 0; i < sizeof(digest); i++)
        snprintf(hex + 2u * i, 3u, "%02x", digest[i]);
    return [NSString stringWithUTF8String:hex];
}

static inline NSString *ds4_metal_binary_sha256(NSString *path) {
    if (!path) return nil;
    FILE *file = fopen([path fileSystemRepresentation], "rb");
    if (!file) return nil;
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    unsigned char buffer[65536];
    size_t n;
    while ((n = fread(buffer, 1, sizeof(buffer), file)) != 0)
        ds4_metal_digest_bytes(&ctx, buffer, n);
    const bool ok = !ferror(file);
    fclose(file);
    return ok ? ds4_metal_digest_finish(&ctx) : nil;
}

#endif
