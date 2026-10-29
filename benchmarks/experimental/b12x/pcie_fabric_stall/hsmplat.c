// Sample HSMP mailbox round-trip latency (HSMP_TEST, read-only) every PERIOD us for SECS seconds.
#include <asm/amd_hsmp.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
static long long now_ns(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
int main(int argc, char** argv) {
  int secs = argc > 1 ? atoi(argv[1]) : 10, period = argc > 2 ? atoi(argv[2]) : 2000;
  int fd = open("/dev/hsmp", O_RDONLY);
  if (fd < 0) { perror("hsmp"); return 1; }
  long long end = now_ns() + (long long)secs * 1000000000LL;
  unsigned v = 0;
  while (now_ns() < end) {
    struct hsmp_message m;
    memset(&m, 0, sizeof m);
    m.msg_id = HSMP_TEST; m.num_args = 1; m.response_sz = 1; m.args[0] = v;
    long long t0 = now_ns();
    int rc = ioctl(fd, HSMP_IOCTL_CMD, &m);
    long long t1 = now_ns();
    printf("%lld %lld %d\n", t0, t1 - t0, rc == 0 && m.args[0] == v + 1);
    ++v;
    usleep(period);
  }
  return 0;
}
