import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    name = 'entry_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 获取文件路径
        self.log_step("step-1", "获取文件路径")
        import_file_path = state['import_file_path']
        # 2. 校验文件路径
        self.log_step("step-2", "校验文件路径")
        if not import_file_path.strip():
            self.logger.error('上传文件路径为空，请重新上传文件后再继续!')
            raise ValidationError('导入文件路径为空，请导入文件后再继续!', self.name)
        import_file_path = Path(import_file_path)
        import_file_suffix = import_file_path.suffix.lower()
        if import_file_suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = str(import_file_path)
        elif import_file_suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = str(import_file_path)
        else:
            self.logger.error("导入的文件后缀必须是pdf或者md，请重新上传文件")
            raise ValidationError('导入的文件后缀必须是pdf或者md，请重新上传文件', self.name)
        file_title = import_file_path.stem
        state['file_title'] = file_title
        return state


if __name__ == '__main__':
    setup_logging()

    test_entry_state = {
        'import_file_path': r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\doc\万用表RS-12的使用.pdf",
        'file_dir': r"G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir"
    }

    entry_node = EntryNode()
    processed_state = entry_node(test_entry_state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))

    """
    返回示例：
    {
        "import_file_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\doc\\万用表RS-12的使用.pdf",
        "file_dir": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\temp_dir",
        "is_pdf_read_enabled": true,
        "pdf_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\doc\\万用表RS-12的使用.pdf",
        "file_title": "万用表RS-12的使用"
    }
    """
