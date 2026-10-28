from enum import Enum
from mammoth.utils.logging import logger


class PatienceEnum(Enum):
    IMPROVING = 0
    DECREASING = 1
    STOPPED = 2


class EarlyStopping(object):
    def __init__(self, tolerance):
        """
        Patience counter for early stopping.

        The improvement signal comes from ModelSaver (--metric_for_best_model,
        averaged over all tasks on all devices), so early stopping and
        best-checkpoint selection always agree.

        Args:
            tolerance(int): number of validations without improvement before stopping
        """

        self.tolerance = tolerance
        self.current_tolerance = self.tolerance
        self.status = PatienceEnum.IMPROVING
        self.current_step_best = 0

    def update(self, is_new_best, step):
        """Update early stopping state using a pre-computed is_new_best boolean.

        :param is_new_best: True if the validation metric improved
        :param step: Current training step
        """
        if self.status == PatienceEnum.STOPPED:
            return

        if is_new_best:
            self.current_step_best = step
            self.current_tolerance = self.tolerance
            self.status = PatienceEnum.IMPROVING
        else:
            self._update_decreasing()

    def _update_decreasing(self):
        # Decrease tolerance
        self.current_tolerance -= 1

        # Log
        logger.info("Decreasing patience: {}/{}".format(self.current_tolerance, self.tolerance))
        # Log
        if self.current_tolerance == 0:
            logger.info("Training finished after not improving. Early Stop!")
            self._log_best_step()

        self._decreasing_or_stopped_status_update(self.current_tolerance)

    def _log_best_step(self):
        logger.info("Best model found at step {}".format(self.current_step_best))

    def _decreasing_or_stopped_status_update(self, tolerance):
        self.status = PatienceEnum.DECREASING if tolerance > 0 else PatienceEnum.STOPPED

    def is_improving(self):
        return self.status == PatienceEnum.IMPROVING

    def has_stopped(self):
        return self.status == PatienceEnum.STOPPED
