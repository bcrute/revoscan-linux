# Scanner-048 accepted — correct zero-attempt preflight stop

The frozen scanner-047 helper was not invoked. All immutable hashes matched, but
read-only preflight found the exact scanner at devnum 50 rather than required
devnum 49, with interfaces 0/1 rebound to `uvcvideo`. Per the assignment, the
executor stopped before creating an evidence directory or opening any scanner
node. No live scanner action or retry occurred.

The host kernel journal later showed that scanner-046's selector and clear
timeouts were followed about 3.4 seconds later by USB device 49 disconnecting;
the same path/VID/PID/serial re-enumerated as device 50 about 14.9 seconds after
that. This proves a post-timeout device-side re-enumeration as observed by the
host, but not whether the cause was the embedded watchdog or some other firmware
reset path. Optical state was not tested and the front emitter remains presumed
on.
