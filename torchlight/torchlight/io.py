#!/usr/bin/env python
"""torchlight 的 I/O 与命令行辅助工具。"""

from __future__ import annotations

import argparse
import atexit
import pickle
import sys
import time
import traceback
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any

from herald import ConsoleHandler
from herald import FileHandler
from herald import LogManager
from herald import get_logger
import numpy as np
import torch
import yaml

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


class IO:
    """轻量 I/O 辅助类。

    该类负责模型实例化、权重读写、参数保存与日志输出。公共方法名与
    旧版保持一致，以确保处理器代码无需调整外部接口。
    """

    _shutdown_registered = False

    def __init__(self, work_dir: str, save_log: bool = True, print_log: bool = True):
        self.work_dir = work_dir
        self.save_log = save_log
        self.print_to_screen = print_log
        self.cur_time = time.time()
        self.split_timer: dict[str, float] = {}
        self.pavi_logger = None
        self.session_file: str | None = None
        self.model_text = ""
        self.logger = self._build_logger()

    def log(self, *args: Any, **kwargs: Any) -> None:
        """兼容旧版接口，当前版本不执行任何实际日志上报。"""

    def _build_logger(self):
        """构建当前 IO 实例使用的 Herald logger。"""
        log_dir = Path(self.work_dir)
        if self.save_log:
            log_dir.mkdir(parents=True, exist_ok=True)

        logger_name = "stgcn_io"
        if log_dir.name:
            logger_name = f"stgcn_io_{log_dir.name}"

        logger = get_logger(logger_name, level="debug")
        if self.print_to_screen:
            logger.add_handler(ConsoleHandler(level="info"))
        if self.save_log:
            logger.add_handler(FileHandler(log_dir / "log.txt", level="debug"))

        if not IO._shutdown_registered:
            atexit.register(LogManager.shutdown)
            IO._shutdown_registered = True

        return logger

    def _emit(self, level: str, message: str) -> None:
        """通过 Herald 按级别输出日志。"""
        getattr(self.logger, level)(message, stacklevel=3)

    def debug(self, message: str) -> None:
        """输出 debug 级别日志。"""
        self._emit("debug", message)

    def info(self, message: str) -> None:
        """输出 info 级别日志。"""
        self._emit("info", message)

    def success(self, message: str) -> None:
        """输出 success 级别日志。"""
        self._emit("success", message)

    def warning(self, message: str) -> None:
        """输出 warning 级别日志。"""
        self._emit("warning", message)

    def error(self, message: str) -> None:
        """输出 error 级别日志。"""
        self._emit("error", message)

    def load_model(self, model: str, **model_args: Any) -> torch.nn.Module:
        """按字符串路径导入并实例化模型。"""
        model_cls = import_class(model)
        model_instance = model_cls(**model_args)
        self.model_text += "\n\n" + str(model_instance)
        return model_instance

    def load_weights(
        self,
        model: torch.nn.Module,
        weights_path: str,
        ignore_weights: str | list[str] | None = None,
    ) -> torch.nn.Module:
        """加载权重文件，保持旧版过滤与兼容加载语义。"""
        if ignore_weights is None:
            ignore_weights = []
        if isinstance(ignore_weights, str):
            ignore_weights = [ignore_weights]

        self.info(f"Load weights from {weights_path}.")
        loaded_weights = torch.load(weights_path, map_location="cpu")
        weights = OrderedDict(
            (key.split("module.")[-1], value.cpu())
            for key, value in loaded_weights.items()
        )

        for ignored_prefix in ignore_weights:
            ignore_name: list[str] = []
            for weight_name in weights:
                if weight_name.find(ignored_prefix) == 0:
                    ignore_name.append(weight_name)
            for name in ignore_name:
                weights.pop(name)
                self.info(
                    "Filter [{}] remove weights [{}].".format(ignored_prefix, name)
                )

        for weight_name in weights:
            self.debug(f"Load weights [{weight_name}].")

        try:
            model.load_state_dict(weights)
        except (KeyError, RuntimeError):
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            for missing_key in diff:
                self.warning(f"Can not find weights [{missing_key}].")
            state.update(weights)
            model.load_state_dict(state)
        return model

    def save_pkl(self, result: Any, filename: str) -> None:
        """保存 pickle 结果。"""
        with Path(self.work_dir, filename).open("wb") as file_obj:
            pickle.dump(result, file_obj)

    def save_h5(self, result: dict[str, Any], filename: str) -> None:
        """保存 HDF5 结果。"""
        with h5py.File(Path(self.work_dir, filename), "w") as file_obj:
            for key, value in result.items():
                file_obj[key] = value

    def save_model(self, model: torch.nn.Module, name: str) -> None:
        """保存模型权重，保持旧版 state_dict 键名处理方式。"""
        model_path = Path(self.work_dir, name)
        state_dict = model.state_dict()
        weights = OrderedDict(
            ("".join(key.split("module.")), value.cpu())
            for key, value in state_dict.items()
        )
        torch.save(weights, model_path)
        self.success(f"The model has been saved as {model_path}.")

    def save_arg(self, arg: argparse.Namespace) -> None:
        """保存最终命令行参数到工作目录。"""
        self.session_file = str(Path(self.work_dir, "config.yaml"))
        arg_dict = vars(arg)
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        with Path(self.session_file).open("w", encoding="utf-8") as file_obj:
            file_obj.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, file_obj, default_flow_style=False, indent=4)

    def print_log(self, log_str: str, print_time: bool = True) -> None:
        """兼容旧版调用，按 info 级别输出日志。"""
        del print_time
        self.info(log_str)

    def init_timer(self, *name: str) -> None:
        """初始化分段计时器。"""
        self.record_time()
        self.split_timer = {key: 0.0000001 for key in name}

    def check_time(self, name: str) -> None:
        """记录一个分段耗时。"""
        self.split_timer[name] += self.split_time()

    def record_time(self) -> float:
        """刷新当前时间戳。"""
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self) -> float:
        """返回并刷新当前分段耗时。"""
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_timer(self) -> None:
        """打印已记录分段的耗时占比。"""
        proportion = {
            key: "{:02d}%".format(
                int(round(value * 100 / sum(self.split_timer.values())))
            )
            for key, value in self.split_timer.items()
        }
        self.info("Time consumption:")
        for key in proportion:
            self.info(
                "\t[{}][{}]: {:.4f}".format(key, proportion[key], self.split_timer[key])
            )


def str2bool(v: str) -> bool:
    """将命令行布尔字面量解析为布尔值。"""
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str2dict(v: str) -> dict[str, Any]:
    """保持旧版字符串字典解析行为。"""
    return eval(f"dict({v})")  # pylint: disable=eval-used


def _import_class_0(name: str) -> Any:
    """兼容旧版保留函数。"""
    components = name.split(".")
    module = __import__(components[0])
    for component in components[1:]:
        module = getattr(module, component)
    return module


def import_class(import_str: str) -> Any:
    """从字符串导入类。"""
    module_str, _sep, class_str = import_str.rpartition(".")
    __import__(module_str)
    try:
        return getattr(sys.modules[module_str], class_str)
    except AttributeError as exc:
        raise ImportError(
            "Class %s cannot be found (%s)"
            % (class_str, traceback.format_exception(*sys.exc_info()))
        ) from exc


class DictAction(argparse.Action):
    """将 `k=v` 风格参数合并到字典中。"""

    def __init__(self, option_strings: list[str], dest: str, nargs: Any = None, **kwargs: Any):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        input_dict = eval(f"dict({values})")  # pylint: disable=eval-used
        output_dict = getattr(namespace, self.dest)
        for key, value in input_dict.items():
            output_dict[key] = value
        setattr(namespace, self.dest, output_dict)
