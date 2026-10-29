// Read-only AMD HSMP queries (GET messages only).
#include <asm/amd_hsmp.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

static int fd;
static int q(int id, int nargs, unsigned a0, int resp, unsigned* out) {
  struct hsmp_message m;
  memset(&m, 0, sizeof m);
  m.msg_id = id; m.num_args = nargs; m.response_sz = resp; m.args[0] = a0; m.sock_ind = 0;
  if (ioctl(fd, HSMP_IOCTL_CMD, &m) < 0) return -1;
  for (int i = 0; i < resp; ++i) out[i] = m.args[i];
  return 0;
}
int main(int argc, char** argv) {
  fd = open("/dev/hsmp", O_RDONLY);
  if (fd < 0) { perror("open /dev/hsmp"); return 1; }
  unsigned o[8];
  if (!q(HSMP_GET_SMU_VER, 0, 0, 1, o)) printf("smu ver %#x\n", o[0]);
  if (!q(HSMP_GET_PROTO_VER, 0, 0, 1, o)) printf("proto %u\n", o[0]);
  if (!q(HSMP_GET_FCLK_MCLK, 0, 0, 2, o)) printf("fclk %u mclk %u\n", o[0], o[1]);
  if (!q(HSMP_GET_C0_PERCENT, 0, 0, 1, o)) printf("c0 %u%%\n", o[0]);
  if (!q(HSMP_GET_SOCKET_POWER, 0, 0, 1, o)) printf("socket power %u mW\n", o[0]);
  for (int n = 0; n < 8; ++n) {
    if (!q(HSMP_GET_NBIO_DPM_LEVEL, 1, (unsigned)n << 16, 1, o))
      printf("nbio %d lclk dpm max %u min %u\n", n, (o[0] >> 8) & 0xff, o[0] & 0xff);
    else printf("nbio %d: error\n", n);
  }
  if (argc > 1) {
    // Sample IO link bandwidth: link ids 0..7, bw type 1 (aggregate).
    int secs = atoi(argv[1]);
    struct timespec t0, t;
    clock_gettime(CLOCK_REALTIME, &t0);
    do {
      clock_gettime(CLOCK_REALTIME, &t);
      printf("%ld.%06ld", t.tv_sec, t.tv_nsec / 1000);
      for (int l = 0; l < 8; ++l) {
        if (!q(HSMP_GET_IOLINK_BANDWITH, 1, ((unsigned)l << 8) | 1, 1, o)) printf(" L%d %u", l, o[0]);
        else printf(" L%d -", l);
      }
      printf("\n");
      usleep(20000);
    } while (t.tv_sec - t0.tv_sec < secs);
  }
  return 0;
}
