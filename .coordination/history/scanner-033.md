# scanner-033 accepted investigation summary

- The owner reported the blue front emitter still on while idle, suspending
  the planned selector-0x12 range-query implementation.
- Static review of the pinned SDK found `0x922=0` generic laser off, `0xb04=0`
  laser-LED disable, and no illumination command in the SDK stream-stop path.
- Root ran each exact disable once with interface 0 detached and no reconnect.
  Both returned 60/60; the owner visually confirmed the blue emitter remained
  on after both. USB acceptance is not shutdown evidence.
- The owner recalls factory software may have the same issue and selected idle
  USB power removal as the durable safety policy. Immediate guidance is to
  unplug until host hub-control permission and the owned implementation exist.
