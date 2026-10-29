// Read-only HSMP SET_GET queries (bit 31 = get). The fd is O_RDONLY, so the driver rejects any SET.
#include <asm/amd_hsmp.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
static int get(int fd, unsigned id, unsigned* out) {
  struct hsmp_message m;
  memset(&m, 0, sizeof m);
  m.msg_id = id; m.num_args = 1; m.response_sz = 1; m.args[0] = 1u << 31;
  if (ioctl(fd, HSMP_IOCTL_CMD, &m) < 0) return -1;
  *out = m.args[0];
  return 0;
}
int main(void) {
  int fd = open("/dev/hsmp", O_RDONLY);
  if (fd < 0) { perror("hsmp"); return 1; }
  const struct { unsigned id; const char* name; } q[] = {
      {0x21, "power efficiency mode"}, {0x27, "cpu rail iso freq policy"}, {0x28, "DF C-state enable"}};
  for (int i = 0; i < 3; ++i) {
    unsigned v = 0;
    if (get(fd, q[i].id, &v)) { printf("%-26s: ", q[i].name); fflush(stdout); perror(""); }
    else printf("%-26s: %#x\n", q[i].name, v);
  }
  return 0;
}
