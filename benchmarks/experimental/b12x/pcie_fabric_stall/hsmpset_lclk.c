// Set the LCLK DPM range of NBIO 0..3 via HSMP: hsmpset_lclk MIN MAX. Prints the range before and after.
#include <asm/amd_hsmp.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
static int msg(int fd, int id, int nargs, unsigned a0, int resp, unsigned* out) {
  struct hsmp_message m;
  memset(&m, 0, sizeof m);
  m.msg_id = id; m.num_args = nargs; m.response_sz = resp; m.args[0] = a0;
  if (ioctl(fd, HSMP_IOCTL_CMD, &m) < 0) return -1;
  if (out) *out = m.args[0];
  return 0;
}
int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s MIN MAX\n", argv[0]); return 2; }
  unsigned lo = atoi(argv[1]), hi = atoi(argv[2]);
  int fd = open("/dev/hsmp", O_RDWR);
  if (fd < 0) { perror("hsmp"); return 1; }
  for (unsigned n = 0; n < 4; ++n) {
    unsigned before = 0, after = 0;
    msg(fd, HSMP_GET_NBIO_DPM_LEVEL, 1, n << 16, 1, &before);
    int rc = msg(fd, HSMP_SET_NBIO_DPM_LEVEL, 1, (n << 16) | (hi << 8) | lo, 0, NULL);
    msg(fd, HSMP_GET_NBIO_DPM_LEVEL, 1, n << 16, 1, &after);
    printf("nbio %u: before max %u min %u -> set rc %d -> after max %u min %u\n", n, (before >> 8) & 0xff,
           before & 0xff, rc, (after >> 8) & 0xff, after & 0xff);
  }
  return 0;
}
