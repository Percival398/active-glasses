// main.c
// Pi Zero real-time grayscale vision pipeline
// OV9281 (Y8) -> remap -> threshold -> morphology
// V4L2 mmap capture (no copies)

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <time.h>

#define W 320
#define H 240
#define N (W * H)

#define CAM_BUFS 4

// ---------------- Buffers ----------------
static uint8_t remap_buf[N];
static uint8_t mask_buf[N];
static uint8_t tmp_buf[N];

static uint16_t remap_x[N];
static uint16_t remap_y[N];

// ---------------- Camera ----------------
static int cam_fd = -1;

struct cam_buffer {
    void   *start;
    size_t  length;
};

static struct cam_buffer cam_bufs[CAM_BUFS];

// ---------------- Timing ----------------
static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// ---------------- Vision ops ----------------
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
        int row = y * W;
        for (int x = 1; x < W - 1; x++) {
            int i = row + x;
            uint8_t m =
                src[i] &
                src[i-1] & src[i+1] &
                src[i-W] & src[i+W] &
                src[i-W-1] & src[i-W+1] &
                src[i+W-1] & src[i+W+1];
            dst[i] = m;
        }
    }
}

static inline void dilate3x3(uint8_t *src, uint8_t *dst)
{
    memset(dst, 0, N);

    for (int y = 1; y < H - 1; y++) {
        int row = y * W;
        for (int x = 1; x < W - 1; x++) {
            int i = row + x;
            uint8_t m =
                src[i] |
                src[i-1] | src[i+1] |
                src[i-W] | src[i+W] |
                src[i-W-1] | src[i-W+1] |
                src[i+W-1] | src[i+W+1];
            dst[i] = m;
        }
    }
}

// ---------------- Remap LUT loading ----------------
static void load_remap(const char *fx, const char *fy)
{
    FILE *f;

    f = fopen(fx, "rb");
    if (!f || fread(remap_x, sizeof(uint16_t), N, f) != N) {
        printf("Using identity remap_x\n");
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                remap_x[y*W + x] = x;
    }
    if (f) fclose(f);

    f = fopen(fy, "rb");
    if (!f || fread(remap_y, sizeof(uint16_t), N, f) != N) {
        printf("Using identity remap_y\n");
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                remap_y[y*W + x] = y;
    }
    if (f) fclose(f);
}

// ---------------- Camera init (mmap) ----------------
static int camera_init(const char *dev)
{
    cam_fd = open(dev, O_RDWR | O_NONBLOCK);
    if (cam_fd < 0) {
        perror("open camera");
        return -1;
    }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = W;
    fmt.fmt.pix.height = H;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(cam_fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("VIDIOC_S_FMT");
        return -1;
    }

    struct v4l2_requestbuffers req = {0};
    req.count = CAM_BUFS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(cam_fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("VIDIOC_REQBUFS");
        return -1;
    }

    for (int i = 0; i < CAM_BUFS; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(cam_fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("VIDIOC_QUERYBUF");
            return -1;
        }

        cam_bufs[i].length = buf.length;
        cam_bufs[i].start = mmap(
            NULL, buf.length,
            PROT_READ | PROT_WRITE,
            MAP_SHARED,
            cam_fd, buf.m.offset
        );

        if (cam_bufs[i].start == MAP_FAILED) {
            perror("mmap");
            return -1;
        }

        if (ioctl(cam_fd, VIDIOC_QBUF, &buf) < 0) {
            perror("VIDIOC_QBUF");
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(cam_fd, VIDIOC_STREAMON, &type) < 0) {
        perror("VIDIOC_STREAMON");
        return -1;
    }

    return 0;
}

// ---------------- Camera grab (DQBUF/QBUF) ----------------
static int camera_grab(uint8_t **out_ptr)
{
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(cam_fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN)
            return -1;
        perror("VIDIOC_DQBUF");
        return -1;
    }

    *out_ptr = (uint8_t *)cam_bufs[buf.index].start;

    if (ioctl(cam_fd, VIDIOC_QBUF, &buf) < 0) {
        perror("VIDIOC_QBUF");
        return -1;
    }

    return 0;
}

// ---------------- MAIN ----------------
int main(void)
{
    load_remap("remap_x.bin", "remap_y.bin");

    if (camera_init("/dev/video0") < 0)
        return 1;

    const uint8_t thresh = 200;

    uint64_t t0, t1;
    int frames = 0;

    while (1) {
        t0 = now_ns();

        uint8_t *frame;
        if (camera_grab(&frame) < 0)
            continue;

        remap_image(frame, remap_buf);
        threshold(remap_buf, mask_buf, thresh);
        erode3x3(mask_buf, tmp_buf);
        dilate3x3(tmp_buf, mask_buf);

        // TODO: LCD DMA transfer of mask_buf

        t1 = now_ns();

        if (++frames % 90 == 0) {
            printf("Frame time: %.2f ms\n", (t1 - t0) / 1e6);
        }
    }

    return 0;
}