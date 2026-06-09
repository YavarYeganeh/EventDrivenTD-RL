"""
Generic command constants used between a controller/trainer and a simulation worker.

These values keep the same protocol as the original code:
    0  initialize
    1  run one interaction step
    2  waiting / completed command
   -1  stop
"""

CMD_INITIALIZE = 0
CMD_RUN_STEP = 1
CMD_WAITING = 2
CMD_STOP = -1
