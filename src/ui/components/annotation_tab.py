"""
Annotation tab component for Gradio UI.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

import gradio as gr

from ...models.annotation import Annotation, Language
from ...annotation.annotation_handler import get_annotation_handler
from ...annotation.statistics import get_statistics
from ...utils.async_helpers import run_async
from ...core.logging import logger


def create_annotation_tab() -> gr.Tab:
    """Create the annotation management tab."""

    with gr.Tab("标注管理") as tab:
        with gr.Row():
            # Left column: List and search
            with gr.Column(scale=2):
                gr.Markdown("### 标注数据列表")

                with gr.Row():
                    search_box = gr.Textbox(
                        label="搜索查询",
                        placeholder="输入关键词搜索...",
                        interactive=True,
                    )
                    search_btn = gr.Button("搜索", variant="secondary")

                with gr.Row():
                    page_size = gr.Dropdown(
                        choices=[10, 20, 50, 100],
                        value=20,
                        label="每页显示",
                    )
                    page_num = gr.Number(value=1, label="页码", precision=0)

                with gr.Row():
                    prev_btn = gr.Button("上一页")
                    next_btn = gr.Button("下一页")

                annotation_list = gr.Dataframe(
                    headers=["ID", "查询", "语言", "FAQ匹配", "创建时间"],
                    datatype=["str", "str", "str", "bool", "str"],
                    col_count=(5, "fixed"),
                    interactive=False,
                    label="标注列表",
                )

                total_count = gr.Markdown("总计: 0 条")

            # Right column: Edit form
            with gr.Column(scale=3):
                gr.Markdown("### 标注详情")

                annotation_id = gr.Textbox(label="ID", interactive=False)
                query_input = gr.Textbox(
                    label="用户查询 *",
                    placeholder="输入用户查询...",
                    lines=2,
                )

                with gr.Row():
                    language_select = gr.Dropdown(
                        choices=[("自动", "auto"), ("中文", "zh"), ("英文", "en")],
                        value="auto",
                        label="语言",
                    )
                    agent_id_input = gr.Textbox(label="Agent ID", value="default")
                    thinking_checkbox = gr.Checkbox(label="开启思考模式", value=False)

                gr.Markdown("#### 多轮对话历史")
                conversation_history = gr.Textbox(
                    label="对话历史（每行一条）",
                    placeholder="输入对话历史，每行一条...",
                    lines=3,
                )

                gr.Markdown("#### 标注结果")

                gt_documents = gr.Textbox(
                    label="Ground Truth 文档",
                    placeholder="输入相关文档，每行一个...",
                    lines=3,
                )

                with gr.Row():
                    faq_matched = gr.Checkbox(label="FAQ 匹配", value=False)
                    should_refuse = gr.Checkbox(label="应拒答", value=False)

                standard_answers = gr.Textbox(
                    label="标准答案（多个答案每行一个）",
                    placeholder="输入标准答案...",
                    lines=3,
                )

                answer_style = gr.Textbox(
                    label="回答风格要求",
                    placeholder="例如：专业、简洁、友好...",
                )

                notes = gr.Textbox(
                    label="备注",
                    placeholder="添加备注...",
                    lines=2,
                )

                custom_fields = gr.JSON(label="自定义扩展字段")

                with gr.Row():
                    save_btn = gr.Button("保存", variant="primary")
                    new_btn = gr.Button("新建")
                    delete_btn = gr.Button("删除", variant="stop")

                status_msg = gr.Markdown("")

        # Event handlers
        async def load_annotations(page: int, size: int, search: str = ""):
            handler = await get_annotation_handler()

            if search:
                result = await handler.search(search, page=page, page_size=size)
            else:
                result = await handler.list(page=page, page_size=size)

            data = []
            for ann in result.items:
                data.append([
                    ann.id[:8] + "...",
                    ann.query[:50] + "..." if len(ann.query) > 50 else ann.query,
                    ann.language.value,
                    ann.faq_matched,
                    ann.created_at.strftime("%Y-%m-%d %H:%M"),
                ])

            return (
                gr.update(value=data),
                f"总计: {result.total} 条 | 第 {page}/{(result.total + size - 1) // size} 页"
            )

        async def load_annotation_detail(selected_data):
            if not selected_data:
                return [gr.update() for _ in range(12)]

            # Get ID from first column
            short_id = selected_data[0]
            # Need to find full ID - this is simplified
            handler = await get_annotation_handler()
            # Search by ID prefix
            all_anns = await handler.list(page=1, page_size=100)
            ann = None
            for a in all_anns.items:
                if a.id.startswith(short_id.rstrip("...")):
                    ann = a
                    break

            if not ann:
                return [gr.update() for _ in range(12)]

            return [
                gr.update(value=ann.id),
                gr.update(value=ann.query),
                gr.update(value=ann.language.value),
                gr.update(value=ann.agent_id),
                gr.update(value=ann.enable_thinking),
                gr.update(value="\n".join(ann.conversation_history)),
                gr.update(value="\n".join(ann.gt_documents)),
                gr.update(value=ann.faq_matched),
                gr.update(value=ann.should_refuse),
                gr.update(value="\n".join(ann.standard_answers)),
                gr.update(value=ann.answer_style),
                gr.update(value=ann.notes),
            ]

        async def save_annotation(
            ann_id, query, language, agent_id, thinking,
            conv_history, gt_docs, faq, refuse, std_answers, style, note
        ):
            if not query:
                return "❌ 错误：查询不能为空"

            handler = await get_annotation_handler()

            try:
                data = {
                    "query": query,
                    "language": Language(language) if language else Language.AUTO,
                    "agent_id": agent_id or "default",
                    "enable_thinking": thinking,
                    "conversation_history": [l.strip() for l in conv_history.split("\n") if l.strip()] if conv_history else [],
                    "gt_documents": [l.strip() for l in gt_docs.split("\n") if l.strip()] if gt_docs else [],
                    "faq_matched": faq,
                    "should_refuse": refuse,
                    "standard_answers": [l.strip() for l in std_answers.split("\n") if l.strip()] if std_answers else [],
                    "answer_style": style or "",
                    "notes": note or "",
                }

                if ann_id:
                    # Update existing
                    await handler.update(ann_id, data)
                    return f"✅ 更新成功: {ann_id}"
                else:
                    # Create new
                    annotation = Annotation(**data)
                    new_id = await handler.create(annotation)
                    return f"✅ 创建成功: {new_id}"

            except Exception as e:
                logger.error(f"Save failed: {e}")
                return f"❌ 保存失败: {str(e)}"

        async def delete_annotation(ann_id):
            if not ann_id:
                return "❌ 请先选择要删除的标注"

            handler = await get_annotation_handler()
            try:
                await handler.delete(ann_id)
                return f"✅ 已删除: {ann_id}"
            except Exception as e:
                return f"❌ 删除失败: {str(e)}"

        def new_annotation():
            return [
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value="auto"),
                gr.update(value="default"),
                gr.update(value=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=False),
                gr.update(value=False),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
            ]

        # Connect events
        search_btn.click(
            fn=lambda p, s, q: run_async(load_annotations(1, s, q)),
            inputs=[page_num, page_size, search_box],
            outputs=[annotation_list, total_count],
        )

        prev_btn.click(
            fn=lambda p, s, q: run_async(load_annotations(max(1, int(p) - 1), s, q)),
            inputs=[page_num, page_size, search_box],
            outputs=[annotation_list, total_count],
        )

        next_btn.click(
            fn=lambda p, s, q: run_async(load_annotations(int(p) + 1, s, q)),
            inputs=[page_num, page_size, search_box],
            outputs=[annotation_list, total_count],
        )

        annotation_list.select(
            fn=lambda s: run_async(load_annotation_detail(s)),
            inputs=[annotation_list],
            outputs=[
                annotation_id, query_input, language_select, agent_id_input,
                thinking_checkbox, conversation_history, gt_documents,
                faq_matched, should_refuse, standard_answers,
                answer_style, notes
            ],
        )

        save_btn.click(
            fn=lambda *args: run_async(save_annotation(*args)),
            inputs=[
                annotation_id, query_input, language_select, agent_id_input,
                thinking_checkbox, conversation_history, gt_documents,
                faq_matched, should_refuse, standard_answers,
                answer_style, notes
            ],
            outputs=[status_msg],
        )

        new_btn.click(
            fn=new_annotation,
            outputs=[
                annotation_id, query_input, language_select, agent_id_input,
                thinking_checkbox, conversation_history, gt_documents,
                faq_matched, should_refuse, standard_answers,
                answer_style, notes, status_msg
            ],
        )

        delete_btn.click(
            fn=lambda aid: run_async(delete_annotation(aid)),
            inputs=[annotation_id],
            outputs=[status_msg],
        )

        # Initial load
        tab.select(
            fn=lambda s: run_async(load_annotations(1, s, "")),
            inputs=[page_size],
            outputs=[annotation_list, total_count],
        )

    return tab