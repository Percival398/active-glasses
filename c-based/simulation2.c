#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define W 320
#define H 240
#define SIZE (W*H)

// ---------------- Buffers ----------------
uint8_t frame[SIZE];
uint8_t remap_buf[SIZE];
uint8_t mask_buf[SIZE];
uint8_t tmp_buf[SIZE];
uint16_t remap_x[SIZE];
uint16_t remap_y[SIZE];

// ---------------- Remap ----------------
void load_dummy_remap() {
    // Identity mapping: remap does nothing
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int i = y * W + x;
            remap_x[i] = x;
            remap_y[i] = y;
        }
    }
}

void remap_image(uint8_t *src, uint8_t *dst) {
    for (int i = 0; i < SIZE; i++) {
        dst[i] = src[remap_y[i] * W + remap_x[i]];
    }
}

// ---------------- Threshold ----------------
void threshold(uint8_t *src, uint8_t *dst, uint8_t t) {
    for (int i = 0; i < SIZE; i++)
        dst[i] = (src[i] > t) ? 255 : 0;
}

// ---------------- Morphology ----------------
void erode3x3(uint8_t *src, uint8_t *dst) {
    memset(dst, 0, SIZE);
    for (int y = 1; y < H-1; y++) {
        for (int x = 1; x < W-1; x++) {
            int i = y*W + x;
            dst[i] =
                src[i] &
                src[i-1] & src[i+1] &
                src[i-W] & src[i+W] &
                src[i-W-1] & src[i-W+1] &
                src[i+W-1] & src[i+W+1];
        }
    }
}

void dilate3x3(uint8_t *src, uint8_t *dst) {
    memset(dst, 0, SIZE);
    for (int y = 1; y < H-1; y++) {
        for (int x = 1; x < W-1; x++) {
            int i = y*W + x;
            dst[i] =
                src[i] |
                src[i-1] | src[i+1] |
                src[i-W] | src[i+W] |
                src[i-W-1] | src[i-W+1] |
                src[i+W-1] | src[i+W+1];
        }
    }
}

// ---------------- Main ----------------
int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: simulation.exe in.raw out.raw\n");
        return 1;
    }

    printf("Simulation pipeline starting\n");

    // Load dummy remap LUTs
    load_dummy_remap();

    // Open input
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { perror("Input file"); return 1; }
    size_t n = fread(frame, 1, SIZE, fin);
    fclose(fin);
    if (n != SIZE) { printf("Error: input size mismatch\n"); return 1; }

    // Processing pipeline
    remap_image(frame, remap_buf);

    const uint8_t thresh = 200;  // example threshold
    threshold(remap_buf, mask_buf, thresh);

    erode3x3(mask_buf, tmp_buf);
    dilate3x3(tmp_buf, mask_buf);

    // Save output
    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { perror("Output file"); return 1; }
    fwrite(mask_buf, 1, SIZE, fout);
    fclose(fout);

    printf("Simulation pipeline done. Output written to %s\n", argv[2]);
    return 0;
}