// Set an HSMP SET_GET policy and read it back: hsmpset_pm power|dfc VALUE.
// power = 0x21 power efficiency mode[2:0]; dfc = 0x28 DF C-state enable[0]. Nothing else is accepted.
#include <asm/amd_hsmp.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
static int msg(int fd, unsigned id, unsigned a0, unsigned* out) {
  struct hsmp_message m;
  memset(&m, 0, sizeof m);
  m.msg_id = id; m.num_args = 1; m.response_sz = 1; m.args[0] = a0;
  if (ioctl(fd, HSMP_IOCTL_CMD, &m) < 0) return -1;
  *out = m.args[0];
  return 0;
}
int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: %s power|dfc VALUE\n", argv[0]); return 2; }
  unsigned id, v = strtoul(argv[2], NULL, 0);
  if (!strcmp(argv[1], "power") && v <= 5) id = 0x21;
  else if (!strcmp(argv[1], "dfc") && v <= 1) id = 0x28;
  else { fprintf(stderr, "refused\n"); return 2; }
  int fd = open("/dev/hsmp", O_RDWR);
  if (fd < 0) { perror("hsmp"); return 1; }
  unsigned before = 0, resp = 0, after = 0;
  int rg = msg(fd, id, 1u << 31, &before);
  int rs = msg(fd, id, v, &resp);
  int ra = msg(fd, id, 1u << 31, &after);
  printf("%s: before %#x (rc %d) -> set %u rc %d resp %#x -> after %#x (rc %d)\n", argv[1], before, rg, v, rs, resp,
         after, ra);
  return (rs || ra || after != v) ? 1 : 0;
}
