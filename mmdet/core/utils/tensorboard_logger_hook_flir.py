from mmcv.runner.hooks.logger.tensorboard import TensorboardLoggerHook
from mmcv.runner.hooks import HOOKS
from mmcv.runner.dist_utils import master_only
from matplotlib.pyplot import Figure


@HOOKS.register_module()
class TensorboardLoggerHookFlir(TensorboardLoggerHook):
    def __init__(self, log_dir=None, interval=10, ignore_last=True, reset_flag=True, by_epoch=True):
        super().__init__(log_dir, interval, ignore_last, reset_flag, by_epoch)
        self.epoch = 1

    def add_epoch(self):
        self.epoch += 1

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            elif isinstance(val, Figure):
                self.writer.add_figure(f"tag_epoch_{self.epoch}", val, global_step=self.epoch)
                self.add_epoch()
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
