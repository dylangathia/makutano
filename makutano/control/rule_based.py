"""Hand-designed controller: serve the phase whose movements need it most.

It is both a comparison strategy and the fall-back if the RL agent fails to
learn. Each phase is scored on the queues it would clear and the longest wait
among its approaches; the pedestrian phase on how many people are waiting and
for how long. The weights are set in advance in the config and reported.
"""


def choose_phase(queues, waits, phases, ped_phase=None, ped_waiting=0, ped_wait_s=0.0,
                 w_queue=1.0, w_wait=0.5, w_ped=5.0):
    """queues: approach -> queue (m); waits: approach -> seconds since served."""
    def score(i):
        if i == ped_phase:
            return w_ped * ped_waiting + w_wait * ped_wait_s if ped_waiting else 0.0
        served = phases[i]
        return (w_queue * sum(queues.get(a, 0.0) for a in served)
                + w_wait * max((waits.get(a, 0.0) for a in served), default=0.0))
    return max(range(len(phases)), key=score)
