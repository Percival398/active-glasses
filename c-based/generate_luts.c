#include <stdio.h>
#include <stdint.h>

#define W 320
#define H 240
#define N (W*H)

int main(void) {
    FILE *fx = fopen("remap_x.bin", "wb");
    FILE *fy = fopen("remap_y.bin", "wb");
    if (!fx || !fy) {
        perror("fopen");
        return 1;
    }

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            uint16_t xv = x;
            uint16_t yv = y;
            fwrite(&xv, sizeof(uint16_t), 1, fx);
            fwrite(&yv, sizeof(uint16_t), 1, fy);
        }
    }

    fclose(fx);
    fclose(fy);

    printf("Dummy remap LUTs generated.\n");
    return 0;
}
