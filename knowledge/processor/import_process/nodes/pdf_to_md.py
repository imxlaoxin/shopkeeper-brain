import json
import subprocess
import time
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMd(BaseNode):
    name = 'pdf_to_md_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        pdf_path, file_dir = self._validate_params(state)
        # 2. 执行mineru并校验执行情况
        self._execute_mineru(pdf_path, file_dir)
        # 3. 获取生成md所在的路径
        md_path = self._get_md_path(pdf_path, file_dir)
        state['md_path'] = md_path
        return state

    def _validate_params(self, state: ImportGraphState) -> Tuple[Path, Path]:
        self.log_step('step-1', '参数校验')
        pdf_path = state['pdf_path']
        file_dir = state['file_dir']
        if not pdf_path.strip():
            self.logger.error('文件路径为空')
            raise ValidationError('文件路径为空', self.name)
        if not file_dir.strip():
            self.logger.info('没有传入输出目录，将给予默认的输出目录')
            default_dir = Path(__file__).parents[1] / 'temp_dir'
            file_dir = str(default_dir)
        pdf_path = Path(pdf_path)
        file_dir = Path(file_dir)
        if not pdf_path.is_file():
            self.logger.error('上传文件不是文件格式')
            raise ValidationError('上传文件不是文件格式', self.name)
        if not file_dir.is_dir():
            self.logger.info('输出目录不存在，将自动进行创建')
            file_dir.mkdir(parents=True, exist_ok=True)

        return pdf_path, file_dir

    def _execute_mineru(self, pdf_path: Path, file_dir: Path) -> None:
        self.log_step('step-2', '执行mineru')
        start_time = time.time()
        proc = subprocess.Popen(
            args=[
                "mineru",
                "-p",
                str(pdf_path),
                "-o",
                str(file_dir),
                "-b",
                "pipeline"
            ],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲，实时输出
        )

        for line in proc.stdout:
            print(line.rstrip())

        ret_code = proc.wait()
        processed_time = time.time() - start_time

        if ret_code != 0:
            self.logger.error('执行mineru失败')
            raise PdfConversionError('执行mineru失败', self.name)

        self.logger.info(f'执行mineru成功，耗时: {processed_time:.2f}，解析文件为: {pdf_path.name}')

    def _get_md_path(self, pdf_path: Path, file_dir: Path) -> str:
        self.log_step('step-3', '获取md路径')
        file_name = pdf_path.stem
        md_path = file_dir / file_name / 'auto' / f'{file_name}.md' # pipeline模式
        # md_path = file_dir / file_name / 'hybrid_auto' / f'{file_name}.md'  # vllm模式
        return str(md_path)


if __name__ == '__main__':
    setup_logging()

    test_entry_state = {
        "import_file_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\doc\\万用表RS-12的使用.pdf",
        "file_dir": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\temp_dir",
        "is_pdf_read_enabled": True,
        "pdf_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\doc\\万用表RS-12的使用.pdf",
        "file_title": "万用表RS-12的使用"
    }

    entry_node = PdfToMd()
    processed_state = entry_node(test_entry_state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))

    """
    返回示例：
    {
        "import_file_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\doc\\万用表RS-12的使用.pdf",
        "file_dir": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\temp_dir",
        "is_pdf_read_enabled": true,
        "pdf_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\doc\\万用表RS-12的使用.pdf",
        "file_title": "万用表RS-12的使用",
        "md_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\temp_dir\\万用表RS-12的使用\\hybrid_auto\\万用表RS-12的使用.md"
    }
    """