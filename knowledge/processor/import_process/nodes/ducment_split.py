import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


@dataclass
class Section:
    title: str  # 章节标题
    parent_title: str  # 章节父标题
    file_title: str  # 文件标题
    body: str  # 完整内容
    part: int | None = None  # 部分编号


@dataclass
class Chunk:
    title: str  # 章节标题
    parent_title: str  # 章节父标题
    file_title: str  # 文件标题
    content: str  # 完整内容(标题 + 正文)
    part: int | None = None  # 部分编号


class DocumentSplitNode(BaseNode):
    name = 'document_split_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 获取并校验输入参数
        md_content, file_title, min_content_len, max_content_len = self._get_and_validate_params(state)
        # 2. 标题切分，组装section
        sections: List[Section] = self._title_split(md_content, file_title)
        # 3. 长切短合
        final_sections: List[Section] = self._long_split_short_combine(sections, min_content_len, max_content_len)
        # 4. 组装chunk
        chunks: List[Chunk] = self._assemble_chunk(final_sections)
        state['chunks'] = [asdict(chunk) for chunk in chunks]
        # 5. 日志备份
        self._log_summary(md_content, chunks, max_content_len)
        self._backup_chunks(state, chunks)
        return state

    def _get_and_validate_params(self, state: ImportGraphState) -> Tuple[str, str, int, int]:
        self.log_step('step-1', '获取并校验输入参数')
        md_content = state['md_content']
        file_title = state['file_title']
        min_content_length = self.config.min_content_length  # 小于最小长度则合并
        max_content_length = self.config.max_content_length  # 大于最大长度则拆分
        if not md_content:
            raise ValidationError('切分的文档内容为空', self.name)
        if not file_title:
            raise ValidationError('文件标题为空', self.name)
        if min_content_length <= 0:
            raise ValidationError('最小内容长度必须大于0', self.name)
        if max_content_length <= 0:
            raise ValidationError('最大内容长度必须大于0', self.name)
        if min_content_length >= max_content_length:
            raise ValidationError('最小内容长度不能大于等于最大内容长度', self.name)

        return md_content, file_title, min_content_length, max_content_length

    def _title_split(self, md_content: str, file_title: str):
        self.log_step('step-2', '标题切分，组装section')
        # 严格限定 Markdown 标题前最多允许 3 个空格
        heading_re = re.compile(r"^( {0,3})(#{1,6})\s+(.+)")
        fence_re = re.compile(r"^\s*(```|~~~)")

        is_fence = False
        md_lines = md_content.split("\n")
        sections: List[Section] = []

        # 存储各级标题内容（1-6级）如：['', '# xxx', '## xxx', '### xxx', '#### xxx', '##### xxx', '###### xxx']
        section_titles = [''] * 7

        cur_section_body_lines = []  # 当前 section 正文内容
        cur_section_title = ''  # 如果开头没有标题，默认为空
        cur_section_title_level = 0  # 0 代表前言/无标题部分

        def _flush():
            section_body = "\n".join(cur_section_body_lines).strip()

            # 只有当正文不为空，或者它是一个明确的标题节点时才生成 Section
            if section_body or cur_section_title:
                # 计算父标题
                cur_section_parent_title = ''
                for i in range(cur_section_title_level - 1, 0, -1):
                    if section_titles[i]:
                        cur_section_parent_title = section_titles[i]
                        break

                # 如果找不到上级 Markdown 标题，则父标题降级使用 file_title
                if not cur_section_parent_title:
                    cur_section_parent_title = file_title

                # 当前 Section 的最终 title
                final_title = cur_section_title if cur_section_title else file_title

                sections.append(Section(
                    title=final_title,
                    parent_title=cur_section_parent_title,
                    file_title=file_title,
                    body=section_body
                ))

        for md_line in md_lines:
            # 判断代码围栏
            if fence_re.match(md_line):
                is_fence = not is_fence

            re_match = heading_re.match(md_line)

            # 非代码块区域且匹配到标准标题
            if not is_fence and re_match:
                _flush()  # 保存上一个 section

                cur_section_title = md_line.strip()  # 保存完整标题行，如 "### 安全手册"
                cur_section_title_level = len(re_match.group(2))

                # 更新标题层级栈
                section_titles[cur_section_title_level] = cur_section_title
                for i in range(cur_section_title_level + 1, 7):
                    section_titles[i] = ''  # 清空更低层级的标题

                cur_section_body_lines = []  # 重置正文行缓存
            else:
                cur_section_body_lines.append(md_line)

        # 循环结束后 flush 最后一个章节
        _flush()

        self.logger.info(f'step-2 结束执行，返回值 --> sections: {sections[:3]}')
        return sections

    def _long_split_short_combine(self, sections: List[Section], min_content_len: int, max_content_len: int):
        self.log_step('step-3', '长切短合')
        splitted_sections: List[Section] = []
        for section in sections:
            # 长切
            splitted_sub_sections: List[Section] = self._long_section_split(section, max_content_len)
            splitted_sections.extend(splitted_sub_sections)
        final_sections: List[Section] = self._short_section_combine(splitted_sections, min_content_len, max_content_len)
        self.logger.info(f'step-3 结束执行，返回值 --> merged_sections: {final_sections[:3]}')
        return final_sections

    def _long_section_split(self, section, max_content_len):
        # 1. 获取section对象属性
        title = section.title
        parent_title = section.parent_title
        file_title = section.file_title
        body = section.body
        # 2. 判断表格
        if '</table>' in body:  # mineru转换后的md中的表格是html表格table格式
            self.logger.info('_long_split 对表格进行降维转译处理.')
            body = MarkdownTableLinearizer.process(body)
            section.body = body  # 必须将降维后的内容写回 section 对象
        # 3. 对标题校验长度，超过50截断
        if len(title) > 50:
            title = title[:50]
        # 4. 拼接标题前缀 (包含 file_title 预估)
        title_prefix = f"{file_title}\n\n{title}\n\n"
        # 5. 计算标题前缀长度 + 内容长度
        total_length = len(title_prefix) + len(body)
        # 6. 判断是否需要切分
        if total_length <= max_content_len:
            return [section]
        # 7. 计算body可用长度，判断长度是否 <= 0
        available_body_length = max_content_len - len(title_prefix)
        if available_body_length <= 0:
            return [section]
        # 8. 进行切分
        split_sections = []
        text_splitter = RecursiveCharacterTextSplitter(  # RecursiveCharacterTextSplitter: 递归字符文本切分器
            chunk_size=available_body_length,  # 单个切分内容最大长度
            chunk_overlap=20,  # 建议保留少量重叠量以保证语义连贯
            keep_separator=True,  # 设为 True，防止句号/换行符丢失
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ", ""]
        )
        texts = text_splitter.split_text(body)
        for idx, text in enumerate(texts):
            new_section = Section(
                title=title,
                parent_title=parent_title,
                file_title=file_title,
                body=text,
                part=idx + 1
            )
            split_sections.append(new_section)
        # self.logger.info(f'长切，返回值 --> split_sections: {split_sections[:3]}')
        return split_sections

    def _short_section_combine(self, current_sections: List[Section], min_content_len: int, max_content_len: int):
        """
        贪心累加算法
        :param splitted_sections:
        :param min_content_len:
        :param max_content_len:
        :return:
        """
        if not current_sections:
            return []

        combined_sections: list[Section] = []
        # 深拷贝或新建对象，避免直接修改输入的 Section
        cur_section = Section(
            title=current_sections[0].title,
            parent_title=current_sections[0].parent_title,
            file_title=current_sections[0].file_title,
            body=current_sections[0].body,
            part=current_sections[0].part
        )

        for next_section in current_sections[1:]:
            # 只要父标题相同，或者当前 section 的 body 为空（说明是纯结构层级），就允许向下合并
            is_same_parent = (cur_section.parent_title == next_section.parent_title) or (not cur_section.body.strip())
            is_too_short = (len(cur_section.body) < min_content_len)

            # 计算合并后的预估长度
            # 合并时，保留原标题，并将被合并小节的标题及内容一起拼接到body中
            merged_body_candidate = f"{cur_section.body.rstrip()}\n\n{next_section.title}\n{next_section.body.lstrip()}"
            is_within_max_limit = (len(cur_section.title) + 2 + len(merged_body_candidate)) <= max_content_len

            # 只有满足：同父标题 + 当前太短 + 合并后不超长，才允许合并
            if is_same_parent and is_too_short and is_within_max_limit:
                cur_section.body = merged_body_candidate
                cur_section.part = None  # 一旦发生合并，清除局部的切片序号标记，避免污染
                # cur_section.title = cur_section.parent_title
            else:
                combined_sections.append(cur_section)
                # 创建新的指针对象
                cur_section = Section(
                    title=next_section.title,
                    parent_title=next_section.parent_title,
                    file_title=next_section.file_title,
                    body=next_section.body,
                    part=next_section.part
                )

        combined_sections.append(cur_section)

        # 4. 对所有 section 的 part 做处理（全局词频统计法）
        # 先统计每个 title 在合并后总共出现的次数
        title_counts = {}
        for sec in combined_sections:
            title_counts[sec.title] = title_counts.get(sec.title, 0) + 1

        title_occurrences = {}
        result = []
        for combined_section in combined_sections:
            original_title = combined_section.title

            # 如果该标题在最终结果中出现了 1 次以上，则进行流水编号
            if title_counts[original_title] > 1:
                title_occurrences[original_title] = title_occurrences.get(original_title, 0) + 1
                new_part = title_occurrences[original_title]

                combined_section.part = new_part
                combined_section.title = f"{original_title}-{new_part}"
            else:
                # 只有唯一的章节，确保不需要 part 后缀
                combined_section.part = None

            result.append(combined_section)

        # self.logger.info(f'短合，返回值 --> merged_sections: {merged_sections[:3]}')
        return combined_sections

    def _assemble_chunk(self, final_sections: List[Section]):
        self.log_step('step-4', '组装chunk')
        chunks = []
        for section in final_sections:
            # 组装时增强 content 语义，便于向量库检索
            header_prefix = f"{section.file_title.rstrip()}\n\n{section.title.rstrip()}\n\n" if section.title != section.file_title else f"# {section.file_title.rstrip()}\n\n"
            full_content = f"{header_prefix}\n{section.body.strip()}"

            chunks.append(Chunk(
                title=section.title,
                parent_title=section.parent_title,
                file_title=section.file_title,
                content=full_content,
                part=section.part if section.part else None
            ))
        return chunks

    def _log_summary(self, raw_content: str, chunks: List[Chunk], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, chunk in enumerate(chunks[:5]):
                title = chunk.title[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, chunks: List[Chunk]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            # local_dir = Path(local_dir, state['file_title'], 'hybrid_auto')
            local_dir = Path(local_dir, state['file_title'], 'auto')
            local_dir.mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(str(local_dir), "chunks.json")

            chunk_dict = [asdict(chunk) for chunk in chunks]

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunk_dict, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()

    file_path = r'G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用_new.md'
    with open(file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    test_state = {
        "file_title": "万用表RS-12的使用",
        "md_content": md_content,
        "file_dir": r'G:\project\python\AI-Model\project\shopkeeper-brain\knowledge\processor\import_process\temp_dir'
    }

    doc_split_node = DocumentSplitNode()
    processed_state = doc_split_node(test_state)

    # doc_split_node.logger.info(json.dumps(processed_state, indent=4, ensure_ascii=False))
