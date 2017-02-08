Run Find_AvgVel_EventDuration to plot windows using the prescribed average
velocity and duration of all the events. This code is used to check these two
parameters used for labeling windows.

```shell
./params/Find_AvgVel_EventDuration \
--avg_vel 5.8 \
--event_durations 6.0 \
--max_mag 1.0
```

`avg_vel` is the chosen average velocity in the region. `event_duration` is the
duration of each event and `max_mag` is the max magnitude of the events to
consider.

Looks like 5.6 is a good avg velocity. 6.0 s duration is fine.
