#include <stdio.h>
#include <stdlib.h>

#define W 320
#define H 240
#define SIZE (W*H)

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: sim in.raw out.raw\n");
        return 1;
    }

    printf("Start\n");

    FILE *in = fopen(argv[1], "rb");
    if (!in) { perror("in"); return 1; }

    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror("out"); return 1; }

    unsigned char buf[SIZE];

    size_t n = fread(buf, 1, SIZE, in);
    printf("Read %zu bytes\n", n);

    fwrite(buf, 1, SIZE, out);

    fclose(in);
    fclose(out);

    printf("Done\n");
    return 0;
}