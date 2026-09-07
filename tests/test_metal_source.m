/* Startup utilities only; links Foundation, never creates a Metal device. */
#include "../ds4_metal_source.h"
#include <assert.h>

static NSString *aggregate(NSString *first, NSString *second) {
    CC_SHA256_CTX ctx;
    ds4_metal_digest_init(&ctx);
    ds4_metal_digest_part(&ctx, @"embedded", [first dataUsingEncoding:NSUTF8StringEncoding]);
    ds4_metal_digest_part(&ctx, @"metal/a.metal", [second dataUsingEncoding:NSUTF8StringEncoding]);
    return ds4_metal_digest_finish(&ctx);
}

int main(void) {
    @autoreleasepool {
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *original_cwd = [fm currentDirectoryPath];
        NSString *root = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [[NSUUID UUID] UUIDString]];
        NSString *bin = [root stringByAppendingPathComponent:@"bin"];
        NSString *cwd = [root stringByAppendingPathComponent:@"cwd"];
        for (NSString *path in @[bin, cwd])
            assert([fm createDirectoryAtPath:[path stringByAppendingPathComponent:@"metal"]
                withIntermediateDirectories:YES attributes:nil error:NULL]);
        NSString *relative = @"metal/a.metal";
        NSString *sidecar = [bin stringByAppendingPathComponent:relative];
        NSString *cwd_source = [cwd stringByAppendingPathComponent:relative];
        NSString *override = [root stringByAppendingPathComponent:@"override.metal"];
        for (NSString *path in @[sidecar, cwd_source, override])
            assert([@"abc" writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
        assert([fm changeCurrentDirectoryPath:cwd]);
        NSString *executable = [bin stringByAppendingPathComponent:@"ds4"];
        ds4_metal_source_origin origin;
        assert([ds4_metal_source_path(executable, relative, NULL, &origin) isEqualToString:sidecar]);
        assert(origin == DS4_METAL_SOURCE_EXECUTABLE);
        assert([ds4_metal_source_path(executable, relative, [override fileSystemRepresentation], &origin) isEqualToString:override]);
        assert(origin == DS4_METAL_SOURCE_OVERRIDE);
        assert([ds4_metal_source_path(executable, relative, "/missing/override", &origin) isEqualToString:sidecar]);
        assert(origin == DS4_METAL_SOURCE_EXECUTABLE);
        assert([fm removeItemAtPath:sidecar error:NULL]);
        assert([ds4_metal_source_path(executable, relative, NULL, &origin) isEqualToString:relative]);
        assert(origin == DS4_METAL_SOURCE_CWD);
        assert([fm removeItemAtPath:cwd_source error:NULL]);
        assert(!ds4_metal_source_path(executable, relative, NULL, &origin));
        assert(origin == DS4_METAL_SOURCE_MISSING);
        assert([ds4_metal_executable_path() isAbsolutePath]);
        assert([fm isReadableFileAtPath:ds4_metal_executable_path()]);
        assert([ds4_metal_binary_sha256(override) isEqualToString:
            @"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"]);
        /* Independent Python hashlib/struct('<Q') vector fixes byte framing. */
        assert([aggregate(@"x\n", @"abc") isEqualToString:@"0f709a49a242b4afbbd80083c6c79290dbcf8258ffe7f356ef6406b405299b18"]);
        assert(![aggregate(@"abc", @"x\n") isEqualToString:aggregate(@"x\n", @"abc")]);
        assert(![aggregate(@"x\n", @"abd") isEqualToString:aggregate(@"x\n", @"abc")]);
        assert([fm changeCurrentDirectoryPath:original_cwd]);
        assert([fm removeItemAtPath:root error:NULL]);
        puts("Metal startup source path/hash tests: PASS (no GPU)");
    }
    return 0;
}
