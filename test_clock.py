from simulation_clock import SimulationClock
from astropy.time import Time, TimeDelta
import time


Time.now() + TimeDelta(1, format="sec")                          # Warm up Astropy's simulated-time arithmetic.

clock = SimulationClock()


print("Starting in realtime mode")
print(clock.get_time())
print()


time.sleep(2)

print("Two real seconds later")
print(clock.get_time())
print()


print("Switching to 10x simulation speed")
clock.set_speed(10)

sim_start = clock.get_time()
real_start = time.monotonic()

print("Simulation start:")
print(sim_start)
print()


time.sleep(2)

sim_end = clock.get_time()
real_end = time.monotonic()

real_elapsed = real_end - real_start
sim_elapsed = (sim_end - sim_start).sec

print("After about 2 real seconds at 10x speed:")
print(sim_end)

print("Real seconds passed:")
print(real_elapsed)

print("Simulated seconds passed:")
print(sim_elapsed)

print("Measured speed:")
print(sim_elapsed / real_elapsed)
print()


print("Switching to 100x simulation speed")
clock.set_speed(100)

sim_start = clock.get_time()
real_start = time.monotonic()

print("Simulation time at speed change:")
print(sim_start)
print()


time.sleep(2)

sim_end = clock.get_time()
real_end = time.monotonic()

real_elapsed = real_end - real_start
sim_elapsed = (sim_end - sim_start).sec

print("After about 2 real seconds at 100x speed:")
print(sim_end)

print("Real seconds passed:")
print(real_elapsed)

print("Simulated seconds passed:")
print(sim_elapsed)

print("Measured speed:")
print(sim_elapsed / real_elapsed)
print()


print("Pausing")
clock.pause()

paused_time_1 = clock.get_time()

print("Paused at:")
print(paused_time_1)
print()


time.sleep(2)

paused_time_2 = clock.get_time()

print("Two real seconds later while paused:")
print(paused_time_2)

print("Simulated seconds passed while paused:")
print((paused_time_2 - paused_time_1).sec)
print()


print("Resuming at 100x")
clock.resume()

sim_start = clock.get_time()
real_start = time.monotonic()

time.sleep(2)

sim_end = clock.get_time()
real_end = time.monotonic()

real_elapsed = real_end - real_start
sim_elapsed = (sim_end - sim_start).sec

print("Two real seconds after resuming:")
print(sim_end)

print("Measured speed:")
print(sim_elapsed / real_elapsed)
print()


print("Switching to 0.5x simulation speed")
clock.set_speed(0.5)

sim_start = clock.get_time()
real_start = time.monotonic()

time.sleep(2)

sim_end = clock.get_time()
real_end = time.monotonic()

real_elapsed = real_end - real_start
sim_elapsed = (sim_end - sim_start).sec

print("After about 2 real seconds at 0.5x speed:")
print(sim_end)

print("Measured speed:")
print(sim_elapsed / real_elapsed)
print()


print("Returning to realtime")
clock.realtime()

print(clock.get_time())