#!/bin/sh
# This process starts in a private user, mount, pid and network namespace.
set -eu
mount --make-rprivate /
job=$1
blender_dir=$2
worker=$3
gpu=$4
root=$job/.root
mount -t tmpfs -o size=64M,nosuid,nodev tmpfs "$root"
mkdir -p "$root"/usr "$root"/lib "$root"/lib64 "$root"/etc "$root"/opt/blender "$root"/dev "$root"/proc "$root"/sys "$root"/input "$root"/out "$root"/tmp
bind_ro() {
    if [ -d "$1" ]; then mkdir -p "$2"; else touch "$2"; fi
    mount --bind "$1" "$2"
    mount -o remount,bind,ro,nosuid "$2"
}
for system in /usr /lib /lib64; do
    if [ -e "$system" ]; then bind_ro "$system" "$root$system"; fi
done
for config in /etc/ld.so.cache /etc/localtime /etc/fonts /etc/OpenCL; do
    if [ -e "$config" ]; then bind_ro "$config" "$root$config"; fi
done
bind_ro "$blender_dir" "$root/opt/blender"
bind_ro "$worker" "$root/input/worker.py"
bind_ro "$job/asset.py" "$root/input/asset.py"
bind_ro "$job/config.json" "$root/input/config.json"
mount --bind "$job/raw" "$root/out"
mount -o remount,bind,rw,nosuid,nodev,noexec "$root/out"
mount -t tmpfs -o size=2G,nosuid,nodev,noexec,mode=1777 tmpfs "$root/tmp"
mount -t proc -o nosuid,nodev,noexec proc "$root/proc"
for device in null zero random urandom; do
    touch "$root/dev/$device"
    mount --bind "/dev/$device" "$root/dev/$device"
done
if [ -n "$gpu" ]; then
    mount -t sysfs -o ro,nosuid,nodev,noexec sysfs "$root/sys"
    for device in "/dev/nvidia$gpu" /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
        touch "$root$device"
        mount --bind "$device" "$root$device"
    done
fi
echo SFT_SANDBOX_READY
set -- /usr/sbin/chroot "$root" /usr/bin/setpriv --no-new-privs --bounding-set=-all --inh-caps=-all --ambient-caps=-all /usr/bin/prlimit --core=0 --fsize=268435456 -- /usr/bin/env HOME=/tmp TMPDIR=/tmp XDG_CACHE_HOME=/tmp SFT_ASSET_PATH=/input/asset.py SFT_CONFIG_PATH=/input/config.json /usr/bin/sh -c 'cd /out; exec /opt/blender/blender --background --threads "$OMP_NUM_THREADS" --factory-startup --disable-autoexec --python-exit-code 12 --python /input/worker.py'




exec "$@"
