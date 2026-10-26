"""Pure observational successful-return progress; no Torch import or RNG draws."""

import math
import time


class Observer:
    def __init__(self, session, backbone, seed, emit, targets=2000, interval=50, seconds=30):
        self.session, self.backbone, self.seed = session, backbone, seed
        self.emit, self.targets, self.interval, self.seconds = emit, targets, interval, seconds
        self.started = time.monotonic()
        self.last_emit, self.epoch, self.count = self.started, None, 0

    def wrap(self, original):
        observer = self

        def observed(instance, epoch, *args, **kwargs):
            result = original(instance, epoch, *args, **kwargs)
            if not math.isfinite(float(result)):
                raise RuntimeError(
                    "Nonfinite returned training loss; no valid progress acknowledgement"
                )
            if observer.epoch != epoch:
                observer.epoch, observer.count = epoch, 0
            observer.count += 1
            now = time.monotonic()
            if (
                observer.count % observer.interval == 0
                or observer.count == observer.targets
                or now - observer.last_emit >= observer.seconds
            ):
                elapsed = now - observer.started
                updates = epoch * observer.targets + observer.count
                attempted = (epoch - getattr(observer, "first_epoch", epoch)) * observer.targets
                if not hasattr(observer, "first_epoch"):
                    observer.first_epoch = epoch
                attempted += observer.count
                observer.emit(
                    {
                        "event": "TARGET_PROGRESS",
                        "session": observer.session,
                        "backbone": observer.backbone,
                        "seed": observer.seed,
                        "epoch": epoch + 1,
                        "target_completed": observer.count,
                        "target_remaining": observer.targets - observer.count,
                        "epoch_percent": 100 * observer.count / observer.targets,
                        "optimizer_updates": updates,
                        "total_updates": 30 * observer.targets,
                        "elapsed_seconds": elapsed,
                        "estimated_remaining_seconds": elapsed
                        / max(1, attempted)
                        * (30 * observer.targets - updates),
                        "status": "TRAINING",
                        "loss": float(result),
                        "progress_scope": "successful optimizer return; not crash-durable",
                    }
                )
                observer.last_emit = now
            return result

        return observed
