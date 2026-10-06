from astropy.time import Time, TimeDelta
import time


class SimulationClock:

    def __init__(self):

        self.mode = "realtime"

        self.sim_time = Time.now()

        self.speed = 1.0
        self.paused = False

        self.real_anchor = time.monotonic()                       # Real-world stopwatch time corresponding to sim_time.


    def get_time(self):

        if self.mode == "realtime":                              # In realtime mode, the universe is the clock.

            return Time.now()

        if self.paused:

            return self.sim_time

        real_seconds_passed = (
            time.monotonic() - self.real_anchor
        )

        sim_seconds_passed = (
            real_seconds_passed * self.speed
        )

        return self.sim_time + TimeDelta(
            sim_seconds_passed,
            format="sec"
        )


    def _capture_current_sim_time(self):

        self.sim_time = self.get_time()                          # Save the current simulated time before changing anything.

        self.real_anchor = time.monotonic()                      # Reset the stopwatch anchor to right now.


    def _enter_simulation_mode(self):

        if self.mode == "realtime":                              # Detach from realtime at the exact current instant.

            self.sim_time = Time.now()

            self.real_anchor = time.monotonic()

            self.mode = "simulation"


    def set_time(self, new_time):

        self.mode = "simulation"                                # Seeking automatically takes the clock out of realtime.

        self.sim_time = Time(new_time)

        self.real_anchor = time.monotonic()


    def set_speed(self, new_speed):

        if self.mode == "realtime":                              # Changing the tick rate automatically takes the clock out of realtime.

            self._enter_simulation_mode()

        else:

            self._capture_current_sim_time()                     # Capture the current time using the OLD speed before changing it.

        self.speed = new_speed


    def pause(self):

        if self.mode == "realtime":                              # Pausing realtime means detaching from reality at this exact instant.

            self._enter_simulation_mode()

        else:

            self._capture_current_sim_time()                     # Capture all elapsed simulated time before freezing it.

        self.paused = True


    def resume(self):

        self.real_anchor = time.monotonic()                      # Start measuring elapsed real time again from right now.

        self.paused = False


    def realtime(self):

        self.mode = "realtime"

        self.sim_time = Time.now()

        self.speed = 1.0
        self.paused = False

        self.real_anchor = time.monotonic()