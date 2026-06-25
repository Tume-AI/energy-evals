#!/usr/bin/env bash
# Network isolation for the energyevals execution sandbox.
#
# Creates a dedicated Docker network and firewalls it so sandbox containers can
# reach the public internet (to fetch public data) but NOT the host's private
# network. Run once with sudo:
#
#     sudo ./scripts/setup_sandbox_network.sh up         # apply now (default)
#     sudo ./scripts/setup_sandbox_network.sh down       # remove rules
#     sudo ./scripts/setup_sandbox_network.sh install    # apply now + re-apply on every boot (systemd)
#     sudo ./scripts/setup_sandbox_network.sh uninstall  # remove the boot service
#     sudo ./scripts/setup_sandbox_network.sh verify     # show service + rules
#
# After 'up'/'install', point the sandbox at the network:
#     echo 'SANDBOX_NETWORK=ee-sandbox' >> .env
set -euo pipefail

NET="ee-sandbox"
SUBNET="172.31.250.0/24"
# Private/internal destinations the sandbox must never reach.
BLOCK=(10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 169.254.0.0/16 100.64.0.0/10)

SERVICE_NAME="energyevals-sandbox-firewall.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
SCRIPT_PATH="$(readlink -f "$0")"

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Run with sudo: sudo $0 ${1:-up}" >&2
    exit 1
  fi
}

ensure_network() {
  if docker network inspect "$NET" >/dev/null 2>&1; then
    echo "docker network '$NET' already exists"
  else
    docker network create --subnet "$SUBNET" "$NET" >/dev/null
    echo "created docker network '$NET' ($SUBNET)"
  fi
}

rule_add() {  # idempotent insert into DOCKER-USER
  iptables -C DOCKER-USER "$@" 2>/dev/null || iptables -I DOCKER-USER "$@"
}

rule_del() {
  while iptables -C DOCKER-USER "$@" 2>/dev/null; do
    iptables -D DOCKER-USER "$@"
  done
}

up() {
  require_root up
  ensure_network
  # Let return traffic of allowed (public) connections back in.
  rule_add -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
  # Drop new connections from the sandbox subnet to private ranges.
  for cidr in "${BLOCK[@]}"; do
    rule_add -s "$SUBNET" -d "$cidr" -j DROP
  done
  echo "firewall installed: sandbox subnet $SUBNET -> private ranges DROPPED, public ALLOWED."
  echo "Next: echo 'SANDBOX_NETWORK=ee-sandbox' >> .env"
}

down() {
  require_root down
  rule_del -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
  for cidr in "${BLOCK[@]}"; do
    rule_del -s "$SUBNET" -d "$cidr" -j DROP
  done
  echo "firewall rules removed (docker network '$NET' left intact; 'docker network rm $NET' to delete)."
}

install_service() {
  require_root install
  cat > "$SERVICE_PATH" <<EOF
[Unit]
Description=energyevals sandbox network egress firewall
# DOCKER-USER rules and the docker network are runtime state that Docker
# recreates on start, so re-apply them on every boot, after Docker is up.
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=${SCRIPT_PATH} up
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
  systemctl daemon-reload
  systemctl enable --now "$SERVICE_NAME"
  echo "installed + enabled $SERVICE_NAME -- firewall now re-applies on every boot."
}

uninstall_service() {
  require_root uninstall
  systemctl disable --now "$SERVICE_NAME" 2>/dev/null || true
  rm -f "$SERVICE_PATH"
  systemctl daemon-reload
  echo "removed $SERVICE_NAME (current rules stay until reboot or '$0 down')."
}

verify() {
  echo "== systemd unit =="
  systemctl is-enabled "$SERVICE_NAME" 2>/dev/null || echo "$SERVICE_NAME not installed"
  echo "== docker network =="
  docker network inspect "$NET" --format '{{.Name}} {{(index .IPAM.Config 0).Subnet}}' 2>/dev/null \
    || echo "network '$NET' not found"
  echo "== DOCKER-USER rules =="
  iptables -S DOCKER-USER 2>/dev/null || echo "(need root to read iptables)"
}

case "${1:-up}" in
  up) up ;;
  down) down ;;
  install) install_service ;;
  uninstall) uninstall_service ;;
  verify) verify ;;
  *) echo "usage: $0 {up|down|install|uninstall|verify}" >&2; exit 2 ;;
esac
