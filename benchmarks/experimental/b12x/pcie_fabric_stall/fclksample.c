// Sample FCLK/MCLK via HSMP (read-only) every PERIOD us for SECS seconds; print changes and a histogram.
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
  unsigned last_f = 0, last_m = 0, n = 0;
  while (now_ns() < end) {
    struct hsmp_message m;
    memset(&m, 0, sizeof m);
    m.msg_id = HSMP_GET_FCLK_MCLK; m.num_args = 0; m.response_sz = 2;
    if (ioctl(fd, HSMP_IOCTL_CMD, &m) == 0) {
      if (m.args[0] != last_f || m.args[1] != last_m) printf("%lld fclk %u mclk %u\n", now_ns(), m.args[0], m.args[1]);
      last_f = m.args[0]; last_m = m.args[1];
      ++n;
    }
    usleep(period);
  }
  printf("# samples %u\n", n);
  return 0;
}
