"""
Annotation tab component for Gradio UI.
Enhanced with modern styling and improved UX.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

import gradio as gr

from ...models.annotation import Annotation, Language
from ...annotation.annotation_handler import get_annotation_handler
from ...annotation.statistics import get_statistics
from ...core.logging import logger

# 全局缓存当前显示的标注ID列表
_current_annotation_ids: list[str] = []


def create_annotation_tab() -> None:
    """Create the annotation management tab with enhanced styling."""
    global _current_annotation_ids

    # Header
    gr.Markdown("""
    ### 📋 标注数据管理
    管理用户查询标注数据，支持搜索、编辑、删除操作
    """)

    with gr.Row():
        # Left column: List and search
        with gr.Column(scale=2):
            # Search section in a card
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**🔍 搜索与分页**")

                with gr.Row():
                    search_box = gr.Textbox(
                        label="搜索查询",
                        placeholder="输入关键词搜索...",
                        interactive=True,
                        scale=3,
                    )
                    search_btn = gr.Button(
                        "🔍 搜索",
                        variant="secondary",
                        scale=1,
                    )

                with gr.Row():
                    page_size = gr.Dropdown(
                        choices=[10, 20, 50, 100],
                        value=20,
                        label="每页显示",
                        scale=1,
                    )
                    page_num = gr.Number(
                        value=1,
                        label="页码",
                        precision=0,
                        scale=1,
                    )

                with gr.Row():
                    prev_btn = gr.Button(
                        "⬅️ 上一页",
                        variant="secondary",
                    )
                    next_btn = gr.Button(
                        "下一页 ➡️",
                        variant="secondary",
                    )

            # Annotation list
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**📄 标注列表**")

                annotation_list = gr.Dataframe(
                    headers=["ID", "查询", "语言", "FAQ 匹配", "创建时间"],
                    datatype=["str", "str", "str", "bool", "str"],
                    interactive=False,
                    label="标注列表",
                    wrap=True,
                )

                total_count = gr.Markdown(
                    "📊 总计：0 条",
                    elem_classes=["markdown-text", "text-muted"],
                )

        # Right column: Edit form and preview
        with gr.Column(scale=3):
            # Form in a card
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**✏️ 标注详情**")

                with gr.Row():
                    with gr.Column(scale=2):
                        annotation_id = gr.Textbox(
                            label="ID",
                            interactive=False,
                            placeholder="选择标注后自动填充",
                        )
                    with gr.Column(scale=1):
                        language_select = gr.Dropdown(
                            choices=[
                                ("🤖 自动", "auto"),
                                ("🇨🇳 中文", "zh"),
                                ("🇺🇸 英文", "en"),
                            ],
                            value="auto",
                            label="语言",
                        )

                query_input = gr.Textbox(
                    label="用户查询 *",
                    placeholder="输入用户查询内容...",
                    lines=2,
                    elem_classes=["required-field"],
                )

                with gr.Row():
                    agent_id_input = gr.Textbox(
                        label="Agent ID",
                        value="default",
                        placeholder="default",
                        scale=2,
                    )
                    thinking_checkbox = gr.Checkbox(
                        label="💭 开启思考模式",
                        value=False,
                        scale=1,
                    )

                with gr.Row():
                    faq_matched = gr.Checkbox(
                        label="✅ FAQ 匹配",
                        value=False,
                    )
                    should_refuse = gr.Checkbox(
                        label="🚫 应拒答",
                        value=False,
                    )

            # 标注结果部分
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**🎯 标注结果**")

                gt_documents = gr.Textbox(
                    label="Ground Truth 文档",
                    placeholder="输入相关文档，每行一个...",
                    lines=3,
                )

                standard_answers = gr.Textbox(
                    label="标准答案",
                    placeholder="输入标准答案，多个答案每行一个...",
                    lines=3,
                )

                with gr.Row():
                    answer_style = gr.Textbox(
                        label="回答风格要求",
                        placeholder="例如：专业、简洁、友好...",
                        scale=2,
                    )
                    notes = gr.Textbox(
                        label="备注",
                        placeholder="添加备注...",
                        scale=1,
                    )

            # 多轮对话和自定义字段
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**💬 多轮对话历史 & 扩展字段**")

                conversation_history = gr.Textbox(
                    label="对话历史",
                    placeholder="输入对话历史，每行一条记录...",
                    lines=3,
                )

                custom_fields = gr.JSON(
                    label="自定义扩展字段",
                    value={},
                )

            # 操作按钮
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**⚡ 操作**")
                with gr.Row():
                    save_btn = gr.Button(
                        "💾 保存",
                        variant="primary",
                        scale=2,
                    )
                    new_btn = gr.Button(
                        "➕ 新建",
                        variant="secondary",
                        scale=1,
                    )
                    delete_btn = gr.Button(
                        "🗑️ 删除",
                        variant="stop",
                        scale=1,
                    )

                status_msg = gr.Markdown(
                    "",
                    elem_classes=["status-message"],
                )

    # ===== Event Handlers =====

    async def load_annotations(page: int, size: int, search: str = ""):
        """Load annotations with pagination and optional search."""
        global _current_annotation_ids
        handler = await get_annotation_handler()

        if search:
            result = await handler.search(search, page=page, page_size=size)
        else:
            result = await handler.list(page=page, page_size=size)

        # 缓存当前显示的标注ID
        _current_annotation_ids = [ann.id for ann in result.items]

        data = []
        for ann in result.items:
            data.append([
                ann.id[:8] + "...",
                ann.query[:50] + "..." if len(ann.query) > 50 else ann.query,
                ann.language.value,
                ann.faq_matched,
                ann.created_at.strftime("%Y-%m-%d %H:%M"),
            ])

        total_pages = max(1, (result.total + size - 1) // size)
        # 确保页码不超过总页数
        page = min(page, total_pages)
        page_info = f"📊 总计：{result.total} 条 | 第 {page}/{total_pages} 页"

        return (
            gr.update(value=data),
            page_info,
            gr.update(value=page),  # 更新页码输入框
        )

    async def load_annotation_detail(evt: gr.SelectData):
        """Load annotation detail when a row is selected."""
        global _current_annotation_ids

        logger.info(f"Select event triggered: index={evt.index}")

        if evt.index is None:
            logger.warning("No row selected")
            return [gr.update() for _ in range(12)]

        # evt.index 是 [row, col] 格式
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        logger.info(f"Selected row index: {row_index}, cached IDs: {len(_current_annotation_ids)}")

        # 使用缓存的ID列表
        if row_index < 0 or row_index >= len(_current_annotation_ids):
            logger.warning(f"Row index {row_index} out of range (cached: {len(_current_annotation_ids)})")
            return [gr.update() for _ in range(12)]

        ann_id = _current_annotation_ids[row_index]
        logger.info(f"Looking up annotation ID: {ann_id}")

        handler = await get_annotation_handler()
        ann = await handler.get(ann_id)

        if not ann:
            logger.warning(f"Annotation not found: {ann_id}")
            return [gr.update() for _ in range(12)]

        logger.info(f"Found annotation: {ann.id}")

        return [
            gr.update(value=ann.id),
            gr.update(value=ann.language.value),
            gr.update(value=ann.query),
            gr.update(value=ann.agent_id),
            gr.update(value=ann.enable_thinking),
            gr.update(value=ann.faq_matched),
            gr.update(value=ann.should_refuse),
            gr.update(value="\n".join(ann.gt_documents or [])),
            gr.update(value="\n".join(ann.standard_answers or [])),
            gr.update(value=ann.answer_style or ""),
            gr.update(value=ann.notes or ""),
            gr.update(value="\n".join(ann.conversation_history or [])),
        ]

    async def save_annotation(
        ann_id, language, query, agent_id, thinking,
        faq, refuse, gt_docs, std_answers, style, note, conv_history
    ):
        """Save annotation (create or update)."""
        if not query:
            return "❌ 错误：查询不能为空"

        handler = await get_annotation_handler()

        try:
            data = {
                "query": query,
                "language": Language(language) if language else Language.AUTO,
                "agent_id": agent_id or "default",
                "enable_thinking": thinking,
                "conversation_history": [
                    l.strip() for l in conv_history.split("\n") if l.strip()
                ] if conv_history else [],
                "gt_documents": [
                    l.strip() for l in gt_docs.split("\n") if l.strip()
                ] if gt_docs else [],
                "faq_matched": faq,
                "should_refuse": refuse,
                "standard_answers": [
                    l.strip() for l in std_answers.split("\n") if l.strip()
                ] if std_answers else [],
                "answer_style": style or "",
                "notes": note or "",
            }

            if ann_id:
                # Update existing
                await handler.update(ann_id, data)
                return f"✅ 更新成功：<code>{ann_id[:8]}...</code>"
            else:
                # Create new
                annotation = Annotation(**data)
                new_id = await handler.create(annotation)
                return f"✅ 创建成功：<code>{new_id[:8]}...</code>"

        except Exception as e:
            logger.error(f"Save failed: {e}")
            return f"❌ 保存失败：{str(e)}"

    async def delete_annotation(ann_id):
        """Delete an annotation."""
        if not ann_id:
            return "❌ 请先选择要删除的标注"

        handler = await get_annotation_handler()
        try:
            await handler.delete(ann_id)
            return f"✅ 已删除：<code>{ann_id[:8]}...</code>"
        except Exception as e:
            return f"❌ 删除失败：{str(e)}"

    def new_annotation():
        """Reset form for new annotation."""
        return [
            gr.update(value=""),  # annotation_id
            gr.update(value="auto"),  # language_select
            gr.update(value=""),  # query_input
            gr.update(value="default"),  # agent_id_input
            gr.update(value=False),  # thinking_checkbox
            gr.update(value=False),  # faq_matched
            gr.update(value=False),  # should_refuse
            gr.update(value=""),  # gt_documents
            gr.update(value=""),  # standard_answers
            gr.update(value=""),  # answer_style
            gr.update(value=""),  # notes
            gr.update(value=""),  # conversation_history
            gr.update(value=""),  # status_msg
        ]

    # ===== Connect Events =====

    async def search_annotations(p, s, q):
        """搜索标注数据"""
        return await load_annotations(1, s, q)

    async def prev_page(p, s, q):
        """上一页"""
        new_page = max(1, int(p) - 1)
        return await load_annotations(new_page, s, q)

    async def next_page(p, s, q):
        """下一页"""
        # 获取总数来计算最大页数
        handler = await get_annotation_handler()
        if q:
            result = await handler.search(q, page=1, page_size=1)
        else:
            result = await handler.list(page=1, page_size=1)

        total_pages = max(1, (result.total + s - 1) // s)
        new_page = min(total_pages, int(p) + 1)
        return await load_annotations(new_page, s, q)

    search_btn.click(
        fn=search_annotations,
        inputs=[page_num, page_size, search_box],
        outputs=[annotation_list, total_count, page_num],
    )

    prev_btn.click(
        fn=prev_page,
        inputs=[page_num, page_size, search_box],
        outputs=[annotation_list, total_count, page_num],
    )

    next_btn.click(
        fn=next_page,
        inputs=[page_num, page_size, search_box],
        outputs=[annotation_list, total_count, page_num],
    )

    # Gradio 6.x: select 事件自动传递 SelectData 对象
    annotation_list.select(
        fn=load_annotation_detail,
        outputs=[
            annotation_id, language_select, query_input, agent_id_input,
            thinking_checkbox, faq_matched, should_refuse, gt_documents,
            standard_answers, answer_style, notes, conversation_history
        ],
    )

    save_btn.click(
        fn=save_annotation,
        inputs=[
            annotation_id, language_select, query_input, agent_id_input,
            thinking_checkbox, faq_matched, should_refuse, gt_documents,
            standard_answers, answer_style, notes, conversation_history
        ],
        outputs=[status_msg],
    )

    new_btn.click(
        fn=new_annotation,
        outputs=[
            annotation_id, language_select, query_input, agent_id_input,
            thinking_checkbox, faq_matched, should_refuse, gt_documents,
            standard_answers, answer_style, notes, conversation_history, status_msg
        ],
    )

    delete_btn.click(
        fn=delete_annotation,
        inputs=[annotation_id],
        outputs=[status_msg],
    )

    # 返回需要初始化加载的组件和函数
    return {
        "annotation_list": annotation_list,
        "total_count": total_count,
        "page_num": page_num,
        "load_annotations": load_annotations,
    }