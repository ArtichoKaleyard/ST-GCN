"""仓库根导入兼容层。

官方仓库历史上通常通过单独安装 `torchlight/` 子包来获得
`import torchlight` 的行为。modern 分支在仓库根引入 `pyproject.toml`
后，需要同时兼容“直接在仓库根运行 main.py”和“作为包安装”这两种入口。
"""

from .torchlight import DictAction
from .torchlight import IO
from .torchlight import import_class
from .torchlight import ngpu
from .torchlight import occupy_gpu
from .torchlight import str2bool
from .torchlight import str2dict
from .torchlight import visible_gpu
