"""
Dataset management tab component for Gradio UI.
Supports dataset CRUD, import/export, and statistics.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import gradio as gr

from ...models.dataset import Dataset, DatasetStatus, DatasetSummary
from ...annotation.dataset_handler import get_dataset_handler
from ...annotation.annotation_handler import get_annotation_handler
from ...core.logging import logger

# Global cache for current dataset IDs
_current_dataset_ids: list[str] = []


def create_dataset_tab() -> dict:
    """Create the dataset management tab."""
    global _current_dataset_ids

    # Header
    gr.Markdown("""
    ### 数据集管理
    管理标注数据集，支持创建、编辑、导入导出和版本管理
    """)

    with gr.Row():
        # Left column: Dataset list
        with gr.Column(scale=2):
            # Dataset list section
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**数据集列表**")

                dataset_list = gr.Dataframe(
                    headers=["名称", "状态", "标注数", "默认", "创建时间"],
                    datatype=["str", "str", "number", "bool", "str"],
                    interactive=False,
                    label="数据集列表",
                    wrap=True,
                )

                with gr.Row():
                    refresh_btn = gr.Button("刷新", variant="secondary")
                    new_dataset_btn = gr.Button("新建数据集", variant="primary")

            # Statistics section
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**数据集统计**")

                stats_display = gr.JSON(
                    label="统计信息",
                    value={},
                )

        # Right column: Edit form
        with gr.Column(scale=3):
            # Dataset details form
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**数据集详情**")

                dataset_id = gr.Textbox(
                    label="ID",
                    interactive=False,
                    placeholder="选择数据集后自动填充",
                )

                name_input = gr.Textbox(
                    label="名称 *",
                    placeholder="输入数据集名称...",
                    interactive=True,
                )

                description_input = gr.Textbox(
                    label="描述",
                    placeholder="输入数据集描述...",
                    lines=2,
                    interactive=True,
                )

                with gr.Row():
                    status_select = gr.Dropdown(
                        choices=[
                            ("草稿", "draft"),
                            ("活跃", "active"),
                            ("归档", "archived"),
                            ("锁定", "locked"),
                        ],
                        value="draft",
                        label="状态",
                    )
                    is_default_checkbox = gr.Checkbox(
                        label="设为默认",
                        value=False,
                    )

                tags_input = gr.Textbox(
                    label="标签",
                    placeholder="输入标签，逗号分隔...",
                    interactive=True,
                )

                version_note_input = gr.Textbox(
                    label="版本说明",
                    placeholder="输入版本说明...",
                    interactive=True,
                )

            # Import/Export section
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**导入导出**")

                with gr.Row():
                    export_btn = gr.Button("导出数据集", variant="secondary")
                    export_file = gr.File(
                        label="导出文件",
                        interactive=False,
                    )

                import_file = gr.File(
                    label="导入文件",
                    file_types=[".json"],
                    interactive=True,
                )

                with gr.Row():
                    import_name = gr.Textbox(
                        label="新名称(可选)",
                        placeholder="留空则使用原名称...",
                    )
                    merge_checkbox = gr.Checkbox(
                        label="合并到同名数据集",
                        value=False,
                    )

                import_btn = gr.Button("导入数据集", variant="primary")
                import_status = gr.Markdown("")

            # Operations section
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**操作**")

                with gr.Row():
                    save_btn = gr.Button("保存", variant="primary", scale=2)
                    delete_btn = gr.Button("删除", variant="stop", scale=1)
                    set_default_btn = gr.Button("设为默认", variant="secondary", scale=1)

                with gr.Row():
                    create_version_btn = gr.Button("创建新版本", variant="secondary")

                status_msg = gr.Markdown("")

    # ===== Event Handlers =====

    async def load_datasets() -> gr.update:
        """Load all datasets."""
        global _current_dataset_ids
        handler = await get_dataset_handler()

        result = await handler.list(page=1, page_size=100)

        # Cache dataset IDs
        _current_dataset_ids = [d.id for d in result.items]

        data = []
        for d in result.items:
            status_map = {
                "draft": "草稿",
                "active": "活跃",
                "archived": "归档",
                "locked": "锁定",
            }
            data.append([
                d.name,
                status_map.get(d.status.value, d.status.value),
                d.annotation_count,
                d.is_default,
                d.created_at.strftime("%Y-%m-%d %H:%M"),
            ])

        return gr.update(value=data)

    async def load_dataset_detail(evt: gr.SelectData):
        """Load dataset detail when selected."""
        global _current_dataset_ids

        if evt.index is None:
            return [gr.update() for _ in range(8)]

        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index

        if row_index < 0 or row_index >= len(_current_dataset_ids):
            return [gr.update() for _ in range(8)]

        dataset_id_val = _current_dataset_ids[row_index]

        handler = await get_dataset_handler()
        dataset = await handler.get(dataset_id_val)

        if not dataset:
            return [gr.update() for _ in range(8)]

        # Load statistics
        stats = await handler.get_statistics(dataset_id_val)

        return [
            gr.update(value=dataset.id),
            gr.update(value=dataset.name),
            gr.update(value=dataset.description),
            gr.update(value=dataset.status.value),
            gr.update(value=dataset.is_default),
            gr.update(value=", ".join(dataset.tags)),
            gr.update(value=dataset.version_note),
            gr.update(value=stats.to_dict()),
        ]

    async def save_dataset(
        ds_id, name, description, status, is_default, tags, version_note
    ):
        """Save dataset (create or update)."""
        if not name:
            return "错误：名称不能为空"

        handler = await get_dataset_handler()

        try:
            tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

            data = {
                "name": name,
                "description": description or "",
                "status": DatasetStatus(status) if status else DatasetStatus.DRAFT,
                "is_default": is_default,
                "tags": tags_list,
                "version_note": version_note or "",
            }

            if ds_id:
                # Update existing
                await handler.update(ds_id, data)
                return f"更新成功：{name}"
            else:
                # Create new
                dataset = Dataset(**data)
                new_id = await handler.create(dataset)
                return f"创建成功：{name}"

        except Exception as e:
            logger.error(f"Save dataset failed: {e}")
            return f"保存失败：{str(e)}"

    async def delete_dataset(ds_id):
        """Delete a dataset."""
        if not ds_id:
            return "请先选择要删除的数据集"

        handler = await get_dataset_handler()
        try:
            await handler.delete(ds_id)
            return f"已删除数据集"
        except Exception as e:
            return f"删除失败：{str(e)}"

    async def set_as_default(ds_id):
        """Set dataset as default."""
        if not ds_id:
            return "请先选择数据集"

        handler = await get_dataset_handler()
        try:
            await handler.set_active(ds_id)
            return "已设为默认数据集"
        except Exception as e:
            return f"设置失败：{str(e)}"

    async def export_dataset_action(ds_id):
        """Export dataset to JSON file."""
        if not ds_id:
            return None, "请先选择要导出的数据集"

        handler = await get_dataset_handler()
        try:
            output_path = await handler.export_dataset(ds_id)
            return output_path, f"导出成功：{output_path.name}"
        except Exception as e:
            return None, f"导出失败：{str(e)}"

    async def import_dataset_action(file, new_name, merge):
        """Import dataset from JSON file."""
        if not file:
            return "", "请选择要导入的文件"

        handler = await get_dataset_handler()
        try:
            file_path = Path(file.name)
            dataset, count = await handler.import_dataset(
                file_path,
                name=new_name if new_name else None,
                merge=merge,
            )
            return "", f"导入成功：{dataset.name}，共 {count} 条标注"
        except Exception as e:
            return "", f"导入失败：{str(e)}"

    async def create_version_action(ds_id, version_note):
        """Create a new version of dataset."""
        if not ds_id:
            return "请先选择数据集"

        handler = await get_dataset_handler()
        try:
            new_version = await handler.create_version(ds_id, version_note or "")
            return f"创建版本成功：{new_version.name}"
        except Exception as e:
            return f"创建版本失败：{str(e)}"

    def new_dataset_form():
        """Reset form for new dataset."""
        return [
            gr.update(value=""),  # dataset_id
            gr.update(value=""),  # name
            gr.update(value=""),  # description
            gr.update(value="draft"),  # status
            gr.update(value=False),  # is_default
            gr.update(value=""),  # tags
            gr.update(value=""),  # version_note
            gr.update(value={}),  # stats
        ]

    # ===== Connect Events =====

    refresh_btn.click(
        fn=load_datasets,
        outputs=[dataset_list],
    )

    new_dataset_btn.click(
        fn=new_dataset_form,
        outputs=[
            dataset_id, name_input, description_input, status_select,
            is_default_checkbox, tags_input, version_note_input, stats_display
        ],
    )

    dataset_list.select(
        fn=load_dataset_detail,
        outputs=[
            dataset_id, name_input, description_input, status_select,
            is_default_checkbox, tags_input, version_note_input, stats_display
        ],
    )

    save_btn.click(
        fn=save_dataset,
        inputs=[
            dataset_id, name_input, description_input, status_select,
            is_default_checkbox, tags_input, version_note_input
        ],
        outputs=[status_msg],
    ).then(
        fn=load_datasets,
        outputs=[dataset_list],
    )

    delete_btn.click(
        fn=delete_dataset,
        inputs=[dataset_id],
        outputs=[status_msg],
    ).then(
        fn=load_datasets,
        outputs=[dataset_list],
    ).then(
        fn=new_dataset_form,
        outputs=[
            dataset_id, name_input, description_input, status_select,
            is_default_checkbox, tags_input, version_note_input, stats_display
        ],
    )

    set_default_btn.click(
        fn=set_as_default,
        inputs=[dataset_id],
        outputs=[status_msg],
    ).then(
        fn=load_datasets,
        outputs=[dataset_list],
    )

    export_btn.click(
        fn=export_dataset_action,
        inputs=[dataset_id],
        outputs=[export_file, status_msg],
    )

    import_btn.click(
        fn=import_dataset_action,
        inputs=[import_file, import_name, merge_checkbox],
        outputs=[import_name, import_status],
    ).then(
        fn=load_datasets,
        outputs=[dataset_list],
    )

    create_version_btn.click(
        fn=create_version_action,
        inputs=[dataset_id, version_note_input],
        outputs=[status_msg],
    ).then(
        fn=load_datasets,
        outputs=[dataset_list],
    )

    # Return components for initialization
    return {
        "dataset_list": dataset_list,
        "load_datasets": load_datasets,
    }