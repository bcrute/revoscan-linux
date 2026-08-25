# The scanner's wireless stack, measured

Status: measured live on this unit on 2026-08-24 through six read-only probes
(scanner-067 through scanner-072), plus one owner-performed over-the-air check
that confirmed the access point is beaconing. Every line quoted below is output
the device produced. Where a claim could not be established, it is listed as not
established rather than inferred.

## The question

The MINI is sold, documented and driven as a USB device. Its init scripts
nonetheless carry a complete wireless stack. That is not by itself evidence of
anything: this firmware also polls `/dev/iio:device1` for a proximity sensor
that does not physically exist on this unit (see
[illumination-actuators.md](illumination-actuators.md)), so referencing a
subsystem in software says nothing about the silicon being present.

The question was therefore whether the wireless stack is dead platform
boilerplate inherited from the RV1109 BSP, or a real and running radio.

It is real, and it is running.

## The radio exists

```
$ ls /sys/class/net
lo
wlan0
wlan1

$ ls /sys/bus/sdio/devices
mmc1:0001:1
mmc1:0001:2
mmc1:0001:3
```

Three SDIO functions enumerated on `mmc1` — the signature of a Broadcom combo
part — with `bcmdhd` bound. Two network interfaces exist, and the driver's
`dhd_dpc`, `dhd_rxf` and `dhd_eventd` kernel threads are running.

## Both interfaces are up, in different modes

```
$ iw dev
phy#0
	Interface wlan1
		addr 12:2c:6b:90:86:84
		ssid MINI-REVO-A2270452
		type AP
		txpower 31.00 dBm
	Interface wlan0
		addr 10:2c:6b:90:86:84
		type managed
		txpower 31.00 dBm
```

`wlan1` is an **access point** whose SSID embeds this unit's own serial
(`A2270452`, from serial `EXAMPLESERIAL0001`). `wlan0` is a station.

```
$ ifconfig wlan1
wlan1     Link encap:Ethernet  HWaddr 12:2C:6B:90:86:84
          inet addr:192.168.179.1  Bcast:192.168.179.255  Mask:255.255.255.0
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
```

A route for that subnet exists (`/proc/net/route` carries `00B3A8C0` =
192.168.179.0/24 on `wlan1`).

## The daemons are running, not merely installed

From the process list:

```
422  dropbear      SSH
450  lighttpd      HTTP
426  fcgiwrap      CGI
565  hostapd       access point
569  zxmain        the vendor scanner application
572  dhcpd         DHCP server
653  udhcpc        DHCP client (wlan0, station side)
```

```
$ netstat -ltn
tcp  0  0  0.0.0.0:80   0.0.0.0:*  LISTEN
tcp  0  0  0.0.0.0:22   0.0.0.0:*  LISTEN
```

Both listen on all interfaces, which includes the AP subnet.

## The live access-point configuration

`hostapd` runs as `hostapd -B /data/hostapd.conf`. The stock
`/etc/hostapd.conf` is **not** the file in force — it is the unmodified
upstream template and still says `ssid=test`. The real configuration:

```
#pop2_init=done
#pop2_enable=enable
interface=wlan1
driver=nl80211
ssid=MINI-REVO-A2270452
channel=0
hw_mode=a
ieee80211n=1
ieee80211ac=1
macaddr_acl=0
wmm_enabled=1
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=Revopoint3d
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP
rsn_pairwise=CCMP
```

Reading that in order: 5 GHz (`hw_mode=a`), 802.11n and 802.11ac enabled,
automatic channel selection, SSID broadcast (not hidden), WPA2-PSK with CCMP
and a fixed factory passphrase.

The two commented `pop2_*` lines are vendor provisioning markers naming a
different Revopoint model, which is consistent with this configuration being
shared across the RV1109 product line rather than authored for the MINI.

## Bandwidth: the depth stream fits

USB 2.0 High Speed is 480 **megabits** per second of signalling — 60 MB/s in
theory, realistically 35–40 MB/s after protocol overhead. The depth stream is
far smaller than that ceiling:

| Stream | Per frame | 15 fps | 30 fps |
| --- | --- | --- | --- |
| `Y16 ` 640x400 depth | 512000 B | 7.68 MB/s = 61.4 Mbit/s | 15.36 MB/s = 122.9 Mbit/s |

A 1x1 802.11ac link on 5 GHz carries a few hundred Mbit/s of real throughput,
so 61 Mbit/s has comfortable margin and even 30 fps is plausible. This is an
arithmetic comparison against a nominal link rate, **not** a measured
throughput; no wireless transfer has been benchmarked.

## The HTTP surface

`lighttpd` serves `/tmp/webroot/` on port 80 with `mod_cgi` and `mod_fastcgi`
available:

```
/tmp/webroot/
  cgi-bin/zx_bt.cgi  zx_cmd.cgi  zx_media.cgi  zx_upload.cgi
  data/ImageL0.bmp  ImageL1.bmp  ImageR0.bmp  ImageR1.bmp
  upload.html  zxmain.log  zxup_progress.log
```

Those four CGI endpoints are the same ones the vendor host SDK
(`lib3DCamera.so`) contains URLs for, so the HTTP path is the SDK's own
network transport rather than something incidental.

The four BMPs are 257080 bytes each — 640x400 plus a BMP header — and are the
raw stereo fringe images. `id_msg_proj_catchframe_t` (`0x485`) exports to
exactly this directory; see [dsp-protocol.md](dsp-protocol.md). The web root is
therefore where the firmware itself puts capture data.

## A second route: the USB gadget supports RNDIS

This host has no wireless adapter, so the HTTP API on 192.168.179.1 cannot be
reached from here over the air. There may be a way to reach the same API over
the cable that is already plugged in.

The device composes its USB personality from a configfs gadget built by
`/etc/init.d/S50usbdevice`. That script supports nine functions, gated by
flags it reads from a configuration file:

```
UMS_EN=off   ADB_EN=off   MTP_EN=off   NTB_EN=off   ACM_EN=off
UAC1_EN=off  UAC2_EN=off  UVC_EN=off   RNDIS_EN=off
```

each switched on by a keyword in `$USB_CONFIG_FILE` — `usb_uvc_en`,
`usb_rndis_en`, and so on — which then compose the gadget's product id and
function list. **`usb_rndis_en` is one of them**: the scanner can, in
principle, present itself as a USB network adapter.

What is currently instantiated is only what this unit uses:

```
$ ls /sys/kernel/config/usb_gadget/*/f*
hid.usb1
uvc.gs1
uvc.gs2

$ ls /sys/class/udc
ffd00000.dwc3
```

Two UVC functions (depth and RGB) and one HID, on a bound DWC3 controller.
No network function is present.

**This has not been attempted and should not be attempted casually.**
Re-composing the gadget means unbinding the UDC, which tears down the UVC
interfaces and forces re-enumeration. If the recomposition is wrong the unit
may not return as a camera, and the recovery path is a power cycle at best.
`$USB_CONFIG_FILE`'s location has not been determined, and whether the kernel
has RNDIS/ECM gadget support compiled in is unconfirmed — `/lib/modules`
contains no `*ether*` module, which means either built-in support or none at
all, and the listing above cannot distinguish those. Establishing that safely
requires reading the kernel config, not writing to configfs.

Recorded as the most promising route to the HTTP API without a wireless
adapter, and as an owner decision, not an action.

## Incidental: the vendor application logs our own control traffic

`/tmp/webroot/zxmain.log` carries lines like:

```
00:51:25.172 uvc_events_proc_xu_g [0569][I] UVC_VC_EXTENSION_UNIT UVC_GET_CUR read data enter ...
00:51:25.172 uvc_events_proc_xu_g [0569][I] read_size=56
```

That is `zxmain` logging the extension-unit control transfers this project
uses as its command channel, with the 56-byte chunk payload size visible. It
is a second observability channel alongside `dmesg`, and it is retrievable over
HTTP from the web root.

## The DHCP scope matches the AP

```
subnet 192.168.179.0 netmask 255.255.255.0 {
    range 192.168.179.100 192.168.179.110;
    option routers 192.168.179.1;
}
```

`dhcpd` is running and its scope matches `wlan1`'s address, so a client that
associates would be addressed automatically. Note that `/etc/dnsmasq.conf`
describes a **different** subnet (`10.201.126.1`) and `dnsmasq` is not running;
the two are inconsistent leftovers, and `dhcpd` is the one in force.

## What is NOT established

- ~~Over-the-air beaconing has not been confirmed.~~ **CONFIRMED 2026-08-24.**
  The owner reports seeing the `MINI-REVO` network in a client's WiFi scan.
  The access point is genuinely beaconing over the air, not merely configured.

  This was the one claim the instruments here could not settle: `wlan1`'s TX
  counter reads 0 because `bcmdhd` offloads beacon transmission to firmware, so
  a zero counter was evidence of nothing either way, and this host has no
  wireless adapter (`/sys/class/net` carries only `enp6s0`, docker bridges and
  veths). A client scan was the only decisive test and an owner performed it.
- **No throughput has been measured.** The bandwidth table above is arithmetic.
- **No client has ever associated.** `iw dev wlan1 station dump` is empty and
  `/proc/net/arp` has no entries.
- **The CGI command surface has not been exercised.** Nothing was sent to any
  endpoint.
- `wlan0` (station) has transmitted 91656 bytes and received none, consistent
  with repeated unanswered association attempts. `/etc/wpa_supplicant.conf`
  holds a hardcoded factory network (`ssid="zhixiang"`, the OEM's name, which
  also appears in `zxmain` and `zx_*.cgi`). No `wpa_supplicant` process is
  running, so that credential is currently inert.

## Security observations

These are properties of the shipped firmware, recorded because they matter to
anyone putting this unit on a network. Nothing here was acted on.

1. The AP passphrase is a fixed string compiled into the vendor's
   configuration. It is not derived from the serial, so it is likely identical
   across units of this model.
2. `dropbear` (SSH) listens on `0.0.0.0:22`, which includes the AP subnet.
3. `lighttpd` serves CGI on `0.0.0.0:80`, including `zx_upload.cgi`, whose
   name suggests a firmware upload path. It was not probed.
4. `/userdata` (`/dev/ubi6_0`, where `/data` points) is mounted read-write, so
   `/data/hostapd.conf` is persistent and modifiable. The root filesystem is
   read-only (`ubi0:rootfs ... ro`).

Changing the passphrase or disabling the AP would be a **persistent**
modification to the device and is deliberately not attempted here.

## Cost: a seventh probe took the device offline

A seventh probe, scanner-073, was written to settle whether the kernel has
RNDIS gadget support. It included:

```
grep -rl USB_CONFIG_FILE /etc /usr/bin >/tmp/o3
```

That is a **recursive grep over a directory of binaries**, and its runtime was
never bounded. The command channel's per-transfer timeout is 1000 ms. The
command outran it, the probe aborted mid-sequence, and its cleanup transfer
failed too.

The consequences, in order:

1. Interfaces 0 and 1 lost their driver. `/dev/video0` and `/dev/video1`
   disappeared; interfaces 2, 3 and 4 were untouched throughout.
2. Reissuing `USBDEVFS_CONNECT` for both interfaces returned 0 and changed
   nothing, because the fault was not on the host side. The kernel log was
   explicit: `uvcvideo` was being probed and the **device** was answering
   `-71 (EPROTO)` to every UVC control query, ending in
   `Failed to initialize the device (-71)`. The scanner's own UVC control
   handler was wedged, left mid-transaction by the aborted command.
3. A USB port reset (`USBDEVFS_RESET`) was issued as the standard remedy. The
   device disconnected (`usb 1-1.1: USB disconnect, device number 72`) and did
   **not** re-enumerate.

The unit is offline and requires a physical unplug/replug. Nothing persistent
was changed: the root filesystem is mounted read-only, every probe wrote only
to `/tmp` (tmpfs, RAM), no firmware was flashed, and a replug has fully
recovered this unit from a comparable state before.

**The lesson, and the fix for any future probe:** every command sent down a
channel with a 1000 ms timeout must have a bounded runtime. `find` over the
whole filesystem was survivable because the earlier probes gave it a 3-second
settle and it completed; `grep -r` over binaries was not. Probe commands should
be validated for bounded runtime the same way they are already validated for
length, read-only-ness and output destination — a check the derivation script
does not currently perform.

## Method

Six read-only probes derived from the reviewed scanner-065 helper, inheriting
its safety envelope unchanged: topology validation, `GETDRIVER`-gated detach of
interface 0, owed-debt cleanup, signal shielding, a closed command allowlist
and a transfer cap. Each probe command was checked to be within the 58-byte
channel limit, to redirect only into the single scratch file, and to be
read-only by inspection as well as by allowlist.

| Probe | sha256 | Question |
| --- | --- | --- |
| scanner-067 | `699ebf56…` | Does a radio exist? |
| scanner-068 | `445cf131…` | Is it configured and serving? |
| scanner-069 | `ef94d049…` | What is actually running? |
| scanner-070 | `3c863086…` | Which config is in force? |
| scanner-071 | `dd2ecf57…` | Is the AP encrypted? |
| scanner-072 | `7efffeeb…` | Can the USB gadget carry a network function? |

All five restored the device cleanly: `final_ok: True` and
`final_interfaces_ok: True` on every run. No command wrote to the device, and
no persistent change was made.
