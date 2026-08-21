import base64
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Set, Literal, Dict, Deque

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from logging import Logger
import json

from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


# ── 数据模型 ──
@dataclass
class ImageContext:
    """图片在 MD 中的上下文信息。"""
    heading: str  # 最近的章节标题
    pre_text: str  # 图片上方的正文内容
    post_text: str  # 图片下方的正文内容


@dataclass
class ImageInfo:
    """一张图片的完整信息。"""
    name: str  # 图片文件名，如 "abc123.jpg"
    path: str  # 图片完整路径
    context: ImageContext  # 在 MD 中的上下文


class MdFileHandler:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def read_md(self, md_path: str) -> Tuple[str, Path, Path]:
        self.logger.info(f'step-1 read_md 开始执行，参数 --> md_path: {md_path}')
        if not md_path.strip():
            self.logger.error('md路径为空')
            raise StateFieldError(self.node_name, 'md_path', str)
        md_path = Path(md_path)
        if not md_path.is_file():
            self.logger.error('md路径非文件')
            raise FileProcessingError('md路径非文件', self.node_name)
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        imgs_dir = md_path.parent / 'images'
        self.logger.info(f'step-1 read_md 结束执行，返回值 --> md_content: {md_content[:30]}, imgs_dir: {imgs_dir}')
        return md_content, imgs_dir, md_path

    def backup(self, md_path: Path, new_md_content: str):
        self.logger.info('step-5 backup 开始执行，参数 --> md_path, new_md_content')
        new_md_path = md_path.with_name(f'{md_path.stem}_new{md_path.suffix}')
        try:
            with open(str(new_md_path), 'w', encoding='utf-8') as f:
                f.write(new_md_content)
        except IOError as e:
            self.logger.error(f'写入新文件失败 {str(new_md_path)}: {e}')
            raise ImageProcessingError(f'写入新文件失败 {str(new_md_path)}: {e}', node_name=self.node_name)
        finally:
            self.logger.info(f'step-5 backup 结束执行，返回值 --> new_md_path: {new_md_path}')
        return str(new_md_path)


class ImageScanner:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def scan_img_dir(self, md_content: str, imgs_dir: Path, image_extensions: Set[str], img_content_length: int) -> \
            List[ImageInfo]:
        self.logger.info(
            f'step-2 scan_img_dir 开始执行，参数 --> md_content: {md_content[:30]}, imgs_dir: {imgs_dir}, image_extensions: {image_extensions}')
        img_infos = []
        for img_file in imgs_dir.iterdir():
            if not img_file.is_file():
                continue
            if img_file.suffix not in image_extensions:
                continue
            img_info: ImageInfo = self._find_context(md_content, img_file, img_content_length)
            if img_info is None:
                self.logger.warning(
                    f"MD文件中未找到图片 {img_file.name} 的引用"
                )
                continue
            img_infos.append(img_info)
        self.logger.info(f'step-2 scan_img_dir 结束执行，返回值 --> img_infos[:3]: {img_infos[:3]}')
        return img_infos

    def _find_context(self, md_content: str, img_file: Path, img_content_length: int) -> ImageInfo:
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(img_file.name) + r".*?\)"
        )
        md_content_lines = md_content.split('\n')
        for line_idx, line_content, in enumerate(md_content_lines):
            if not pattern.search(line_content):
                continue
            prev_heading, prev_boundary_idx = self._find_heading_above(line_idx, md_content_lines)
            next_boundary_idx = self._find_heading_below(line_idx, md_content_lines)
            pre_text = self._extract_limited_context(md_content_lines[prev_boundary_idx + 1:line_idx],
                                                     img_content_length, direction='above')
            post_text = self._extract_limited_context(md_content_lines[line_idx + 1:next_boundary_idx],
                                                      img_content_length, direction='below')
            # self.logger.info(f'cur_img_line_idx: {line_idx}, prev_heading: {prev_heading}, prev_boundary_idx: {prev_boundary_idx}, next_boundary_idx: {next_boundary_idx}, pre_text: {pre_text}, post_text: {post_text}')
            img_info = ImageInfo(
                name=img_file.name,
                path=str(img_file),
                context=ImageContext(
                    heading=prev_heading,
                    pre_text=pre_text,
                    post_text=post_text
                )
            )
            return img_info

    def _find_heading_above(self, cur_img_line_idx: int, md_content_lines: List[str]) -> Tuple[str, int]:
        """从 cur_img_line_idx 向上查找最近的标题。"""
        for i in range(cur_img_line_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_content_lines[i]):
                return md_content_lines[i], i
        return "", -1

    def _find_heading_below(self, cur_img_line_idx, md_content_lines):
        """从 cur_img_line_idx 向下查找最近的标题。"""
        for i in range(cur_img_line_idx + 1, len(md_content_lines)):
            if re.match(r"^#{1,6}\s+", md_content_lines[i]):
                return i
        return len(md_content_lines)

    def _extract_limited_context(self, content_lines: List[str], img_content_length: int,
                                 direction: Literal['above', 'below']) -> str:
        """按段落分割，按 direction 方向贪心装填，保持段落完整性。"""
        current_paragraph: List[str] = []
        paragraphs: List[str] = []

        for line in content_lines:
            # line.strip(): 去除字符串首尾的空白字符(空格、制表符、换行符等)
            is_blank_line = not line.strip()
            is_other_image = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue

            current_paragraph.append(line)

        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        if direction == "front":
            paragraphs.reverse()  # 就近原则

        total = 0
        selected: List[str] = []
        for para in paragraphs:
            if (total + len(para) > img_content_length) and selected:  # 至少有个段落
                break
            selected.append(para)
            total += len(para)

        if direction == "front":
            selected.reverse()  # 与原文顺序一致，利于VLM

        return "\n\n".join(selected)  # 折行并空一行


class VLMSummarizer:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def summarize_all(self, img_infos: List[ImageInfo], document_title: str, vl_model: str, requests_per_minute: int) -> \
            Dict[str, str]:
        self.logger.info(
            f'step-3 summarize_all 开始执行，参数 --> img_infos, vl_model: {vl_model}, requests_per_minute: {requests_per_minute}')
        imgs_summarize_map: Dict[str, str] = {}
        req_timestamps: Deque[float] = deque()
        try:
            vlm_client = AIClients.get_openai()
            for img_info in img_infos:
                # 滑动窗口限流
                self._enforce_rate_limit(req_timestamps, requests_per_minute)
                img_summary = self._summarize_one(img_info, document_title, vlm_client, vl_model)
                imgs_summarize_map[img_info.name] = img_summary
        except Exception as e:
            self.logger.warning(
                f"VLM 不可用，跳过图片摘要生成: {e}"
            )
            for img_info in img_infos:
                imgs_summarize_map[img_info.name] = '图片描述'
            return imgs_summarize_map
        finally:
            self.logger.info(f'step-3 summarize_all 结束执行，返回值 --> imgs_summarize_map: {imgs_summarize_map}')
        return imgs_summarize_map

    def _summarize_one(self, img_info, document_title, vlm_client, vl_model):
        parts = [p for p in (
            img_info.context.heading,
            img_info.context.pre_text,
            img_info.context.post_text
        ) if p]
        final_context = "\n".join(parts) if parts else "暂无可用上下文"
        try:
            with open(img_info.path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return "暂无图片"
        try:
            completion = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_title}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }],
            )
        except Exception:
            self.logger.warning('单张图片分析失败，将保留本地路径')
            return '图片描述'
        return completion.choices[0].message.content

    def _enforce_rate_limit(
            self, req_timestamps: Deque[float],
            max_requests: int, window: int = 60,    # 60秒窗口
    ):
        now = time.time()
        while req_timestamps and now - req_timestamps[0] >= window:
            req_timestamps.popleft()

        if len(req_timestamps) >= max_requests:
            sleep_dur = window - (now - req_timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while req_timestamps and now - req_timestamps[0] >= window:
                req_timestamps.popleft()

        req_timestamps.append(now)


class ImageUploader:
    def __init__(self, logger: Logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def upload_and_replace(self, img_infos: List[ImageInfo], imgs_summarize_map: Dict[str, str], document_title: str,
                           md_content: str,
                           minio_base_url: str,
                           minio_bucket: str):
        self.logger.info(
            f'step-4 upload_and_replace 开始执行，参数 --> img_infos, imgs_summarize_map: {imgs_summarize_map}')
        remote_urls_map = self._upload_all(img_infos, document_title, minio_base_url, minio_bucket)
        new_md_content = self._replace_in_md(md_content, remote_urls_map, imgs_summarize_map)
        self.logger.info(f'step-4 upload_and_replace 结束执行，返回值 --> new_md_content: {new_md_content[:50]}')
        return new_md_content

    def _upload_all(self, img_infos: List[ImageInfo], document_title: str, minio_base_url, minio_bucket: str):
        remote_urls_map = {}
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.warning('获取minio客户端失败，所有图片将保留本地路径')
            for img_info in img_infos:
                remote_urls_map[img_info.name] = img_info.path
            return remote_urls_map
        for img_info in img_infos:
            try:
                object_name = f"{document_title}/{img_info.name}"
                minio_client.fput_object(
                    bucket_name=minio_bucket,
                    object_name=object_name,
                    file_path=img_info.path,
                )
                remote_url = f'{minio_base_url}/{minio_bucket}/{object_name}'
                remote_urls_map[img_info.name] = remote_url
            except Exception as e:
                self.logger.warning(f'{img_info.name} 上传失败，保留本地路径')
                remote_urls_map[img_info.name] = img_info.path
        return remote_urls_map

    def _replace_in_md(self, md_content: str, remote_urls_map: Dict[str, str],
                       imgs_summarize_map: Dict[str, str]) -> str:
        """替换 MD 中的图片引用为远程 URL + 摘要。"""
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            original_path = match.group(2).strip()
            file_name_in_md = Path(original_path).name
            for img_name, summary in imgs_summarize_map.items():
                if img_name == file_name_in_md:
                    return f"![{summary}]({remote_urls_map[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)


class MarkDownImageNode(BaseNode):
    name = 'md_img_node'

    def __init__(self):
        super().__init__()
        self.md_file_handler = MdFileHandler(self.logger, self.name)
        self.img_scanner = ImageScanner(self.logger, self.name)
        self.vlm_summarizer = VLMSummarizer(self.logger, self.name)
        self.img_uploader = ImageUploader(self.logger, self.name)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 读取md内容并且根据md路径，构建其images所在目录路径
        self.log_step('step-1', '读取md内容并且根据md路径，构建其images所在目录路径')
        md_content, imgs_dir, md_path = self.md_file_handler.read_md(state['md_path'])
        # 2. 读取images目录，构建图片上下文
        self.log_step('step-2', '读取images目录，构建图片上下文')
        img_infos: List[ImageInfo] = self.img_scanner.scan_img_dir(md_content, imgs_dir, self.config.image_extensions,
                                                                   self.config.img_content_length)
        # 3. 调用vlm获取图片总结
        self.log_step('step-3', '调用vlm获取图片总结')
        document_title = md_path.stem
        imgs_summarize_map: Dict[str, str] = self.vlm_summarizer.summarize_all(img_infos, document_title,
                                                                               self.config.vl_model,
                                                                               self.config.requests_per_minute)
        # 4. 上传图片获取url并进行md内容的替换
        self.log_step('step-4', '上传图片获取url并进行md内容的替换')
        new_md_content = self.img_uploader.upload_and_replace(img_infos,
                                                              imgs_summarize_map,
                                                              document_title,
                                                              md_content,
                                                              minio_base_url=self.config.get_minio_base_url(),
                                                              minio_bucket=self.config.minio_bucket)

        # 5. 写入新的md文件(查看效果)
        self.log_step('step-5', '写入新的md文件(查看效果)')
        self.md_file_handler.backup(md_path, new_md_content)

        # 更新state md_content并返回
        state['md_content'] = new_md_content
        return state


if __name__ == '__main__':
    setup_logging()

    test_state = {
        "md_path": "G:\\project\\python\\AI-Model\\project\\shopkeeper-brain\\knowledge\\processor\\import_process\\temp_dir\\万用表RS-12的使用\\hybrid_auto\\万用表RS-12的使用.md"
    }

    md_img_node = MarkDownImageNode()
    processed_state = md_img_node(test_state)

    md_img_node.logger.info(json.dumps(processed_state, indent=4, ensure_ascii=False))
