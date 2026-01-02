// main_win.c
// Windows simulation of Pi Zero vision pipeline
// Uses raw grayscale frames or BMP input

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define W 320
#define H 240
#define N (W * H)

// ---------------- Buffers ----------------
static uint8_t cam_buf[N];
static uint8_t remap_buf[N];
static uint8_t mask_buf[N];
static uint8_t tmp_buf[N];

static uint16_t remap_x[N];
static uint16_t remap_y[N];

// ---------------- Timing ----------------
static inline double now_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1000.0 / freq.QuadPart;
#else
    return 0.0;
#endif
}

// ---------------- Vision ops (IDENTICAL) ----------------
static inline void remap_image(uint8_t *src, uint8_t *dst)
{
    for (int i = 0; i < N; i++) {
        dst[i] = src[ remap_y[i] * W + remap_x[i] ];
    }
}

static inline void threshold(uint8_t *src, uint8_t *dst, uint8_t t)
{
    for (int i = 0; i < N; i++) {
        dst[i] = (src[i] > t) ? 255 : 0;
    }
}

static inline void erode3x3(uint8_t *src, uint8_t *dst)
{
    memset(dst, 0, N);
    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            int i = y * W + x;
            dst[i] =
                src[i] &
                src[i-1] & src[i+1] &
                src[i-W] & src[i+W] &
                src[i-W-1] & src[i-W+1] &
                src[i+W-1] & src[i+W+1];
        }
    }
}

static inline void dilate3x3(uint8_t *src, uint8_t *dst)
{
    memset(dst, 0, N);
    for (int y = 1; y < H - 1; y++) {
        for (int x = 1; x < W - 1; x++) {
            int i = y * W + x;
            dst[i] =
                src[i] |
                src[i-1] | src[i+1] |
                src[i-W] | src[i+W] |
                src[i-W-1] | src[i-W+1] |
                src[i+W-1] | src[i+W+1];
        }
    }
}

// ---------------- Remap LUT ----------------
static void load_remap(const char *fx, const char *fy)
{
    FILE *f = fopen(fx, "rb");
    if (!f || fread(remap_x, sizeof(uint16_t), N, f) != N) {
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                remap_x[y*W + x] = x;
    }
    if (f) fclose(f);

    f = fopen(fy, "rb");
    if (!f || fread(remap_y, sizeof(uint16_t), N, f) != N) {
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                remap_y[y*W + x] = y;
    }
    if (f) fclose(f);
}

// ---------------- Frame source ----------------
static int load_raw_frame(const char *path, uint8_t *dst)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fread(dst, 1, N, f);
    fclose(f);
    return 0;
}

// ---------------- Main ----------------
int main(void)
{
    load_remap("remap_x.bin", "remap_y.bin");

    const uint8_t thresh = 200;

    double t0, t1;

    // Loop through frames frame_000.raw, frame_001.raw, ...
    for (int frame = 0; frame < 1000; frame++) {
        char fname[256];
        sprintf(fname, "frames/frame_%03d.raw", frame);

        if (load_raw_frame(fname, cam_buf) < 0)
            break;

        t0 = now_ms();

        remap_image(cam_buf, remap_buf);
        threshold(remap_buf, mask_buf, thresh);
        erode3x3(mask_buf, tmp_buf);
        dilate3x3(tmp_buf, mask_buf);

        t1 = now_ms();
        printf("Frame %d: %.3f ms\n", frame, t1 - t0);

        // Optional: dump mask to file for inspection
        // fwrite(mask_buf, 1, N, stdout);
    }

    return 0;
}