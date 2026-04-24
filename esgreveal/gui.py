from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable
import io
import json
import shutil
import subprocess
import sys
import time
import traceback

from .knowledge_base import KnowledgeBaseBuild, build_knowledge_base, knowledge_base_to_json
from .llm import LLMConfig, cli_is_available, default_reasoning_effort_for_model, suggested_models_for_backend
from .pipeline import (
    PipelineLLMError,
    PipelineRun,
    batch_results_to_json,
    results_to_json,
    results_to_markdown,
    run_batch,
    run_pipeline,
)
from .runtime_env import configure_runtime_environment


class _WorkerAPI:
    def __init__(self, signal: Any) -> None:
        self._signal = signal

    def log(self, message: str) -> None:
        self._signal.emit({"kind": "log", "message": message})

    def progress(self, value: int, total: int, message: str = "") -> None:
        self._signal.emit({"kind": "progress", "value": value, "total": total, "message": message})


_MULTI_INPUT_SEPARATOR = " || "


def _parse_input_targets(raw: str) -> list[Path]:
    value = raw.strip()
    if not value:
        return []
    if _MULTI_INPUT_SEPARATOR in value:
        parts = [part.strip() for part in value.split(_MULTI_INPUT_SEPARATOR) if part.strip()]
        return [Path(part) for part in parts]
    return [Path(value)]


def _format_input_targets(paths: list[Path]) -> str:
    return _MULTI_INPUT_SEPARATOR.join(str(path) for path in paths)


def _build_pdf_viewer():
    """
    PDF 뷰어 위젯과 로드 함수를 반환한다.
    우선순위: QPdfView (PySide6.QtPdf) → QWebEngineView → QLabel fallback
    반환: (widget, load_fn)  — load_fn(path: str | Path) 호출 시 PDF 표시
    """
    from pathlib import Path as _Path

    # 1) QPdfView — 가장 경량, 외부 의존 없음
    try:
        from PySide6.QtPdf import QPdfDocument
        from PySide6.QtPdfWidgets import QPdfView
        from PySide6.QtWidgets import QVBoxLayout, QWidget

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        pdf_doc = QPdfDocument(container)
        pdf_view = QPdfView(container)
        pdf_view.setDocument(pdf_doc)
        pdf_view.setPageMode(QPdfView.PageMode.MultiPage)
        vbox.addWidget(pdf_view)

        def _load(path: str) -> None:
            pdf_doc.load(str(_Path(path)))

        return container, _load
    except Exception:
        pass

    # 2) QWebEngineView — Chromium 내장 PDF 렌더러
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtWebEngineWidgets import QWebEngineView

        view = QWebEngineView()

        def _load(path: str) -> None:
            view.load(QUrl.fromLocalFile(str(_Path(path).resolve())))

        return view, _load
    except Exception:
        pass

    # 3) fallback — 경로만 표시
    from PySide6.QtWidgets import QLabel
    label = QLabel("PDF 뷰어를 사용하려면\nPySide6.QtPdf 또는 QtWebEngine이 필요합니다.")
    label.setWordWrap(True)

    def _load(path: str) -> None:
        label.setText(f"PDF: {path}\n\n(뷰어 미설치 — 파일을 직접 여세요)")

    return label, _load


def _monospace_font():
    from PySide6.QtGui import QFont
    font = QFont("Consolas")
    if not font.exactMatch():
        font = QFont("Courier New")
    font.setPointSize(10)
    return font


def _format_prompt_readable(prompt: str) -> str:
    """LLM 프롬프트를 섹션 구분선으로 가독성 있게 포맷한다."""
    import re as _re
    # [SectionName] 헤더 앞에 구분선 삽입
    result = _re.sub(
        r"(?m)^(\[.+?\])\s*$",
        lambda m: "\n" + "═" * 52 + "\n" + m.group(1) + "\n" + "─" * 52,
        prompt,
    )
    return result.lstrip("\n")


def launch_gui() -> int:
    configure_runtime_environment()
    try:
        from PySide6.QtCore import QThread, Qt, Signal
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QInputDialog,
            QLabel,
            QLineEdit,
            QListWidget,
            QListWidgetItem,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QPlainTextEdit,
            QProgressBar,
            QScrollArea,
            QSplitter,
            QSpinBox,
            QTableWidget,
            QTableWidgetItem,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:
        raise RuntimeError("PySide6가 설치되지 않아 GUI를 시작할 수 없습니다.") from exc

    icon_path = Path(__file__).resolve().parent.parent / "assets" / "esgreveal_icon.ico"

    class TaskThread(QThread):
        progress = Signal(dict)
        finished_ok = Signal(object)
        failed = Signal(object)

        def __init__(self, fn: Callable[..., Any], kwargs: dict[str, Any]) -> None:
            super().__init__()
            self._fn = fn
            self._kwargs = kwargs

        def run(self) -> None:
            # transformers>=5.5 의 loading_report.py 가 sys.stdout.isatty() 를 호출하는데
            # QThread 에서 sys.stdout 이 None 이면 AttributeError 가 발생한다.
            _patched = sys.stdout is None
            if _patched:
                sys.stdout = io.StringIO()
            try:
                result = self._fn(worker=_WorkerAPI(self.progress), **self._kwargs)
            except Exception:
                self.failed.emit({"traceback": traceback.format_exc(), "exception": sys.exc_info()[1]})
                return
            finally:
                if _patched:
                    sys.stdout = None
            self.finished_ok.emit(result)

    def build_kb_task(*, worker: _WorkerAPI, **kwargs: Any) -> KnowledgeBaseBuild:
        worker.log("Knowledge Base 구축을 시작합니다.")
        build = build_knowledge_base(**kwargs, worker=worker)
        worker.log("Knowledge Base 구축이 끝났습니다.")
        return build

    def run_extract_task(*, worker: _WorkerAPI, **kwargs: Any) -> list[PipelineRun]:
        input_value = kwargs["input_path"]
        input_targets = [Path(path) for path in input_value] if isinstance(input_value, (list, tuple)) else [Path(input_value)]
        worker.log("추출을 시작합니다.")
        if len(input_targets) == 1 and input_targets[0].is_dir():
            input_path = input_targets[0]
            runs = run_batch(
                input_path=input_path,
                metadata_path=kwargs.get("metadata_path"),
                top_k=kwargs["top_k"],
                recursive=kwargs["recursive"],
                llm_config=kwargs.get("llm_config"),
                kb_dir=kwargs.get("kb_dir"),
                retrieval_initial_k=kwargs["initial_k"],
                worker=worker,
            )
            worker.log(f"{len(runs)}개 문서 추출이 끝났습니다.")
            return runs

        runs = []
        total = max(len(input_targets), 1)
        for index, input_path in enumerate(input_targets, start=1):
            worker.log(f"[{index}/{total}] {input_path.name} 추출 중...")
            runs.append(
                run_pipeline(
                    pdf_path=input_path,
                    metadata_path=kwargs.get("metadata_path"),
                    top_k=kwargs["top_k"],
                    llm_config=kwargs.get("llm_config"),
                    kb_dir=kwargs.get("kb_dir"),
                    retrieval_initial_k=kwargs["initial_k"],
                    worker=worker,
                )
            )
        worker.log(f"{len(runs)}개 문서 추출이 끝났습니다.")
        return runs

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("ESGReveal Workbench")
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
            self.resize(1600, 980)
            self._thread: TaskThread | None = None
            self._runs: list[PipelineRun] = []
            self._current_run: PipelineRun | None = None
            self._gemini_install_prompted = False
            self._task_start_time: float = 0.0
            self._dark_mode: bool = False
            self._last_extract_kwargs: dict[str, Any] | None = None

            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            top = QSplitter(Qt.Horizontal)
            top.setChildrenCollapsible(False)
            layout.addWidget(top, stretch=1)

            controls_scroll = QScrollArea()
            controls_scroll.setMinimumWidth(480)
            controls_scroll.setWidgetResizable(True)
            controls_scroll.setFrameShape(QScrollArea.NoFrame)
            controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            controls_widget = self._build_controls()
            controls_widget.setMinimumWidth(460)
            controls_scroll.setWidget(controls_widget)

            top.addWidget(controls_scroll)
            top.addWidget(self._build_results())
            top.setSizes([560, 1040])

            self.status_label = QLabel("대기 중")
            self.progress = QProgressBar()
            self.progress.setRange(0, 0)
            self.progress.hide()
            self.eta_label = QLabel("")
            self.eta_label.hide()
            self.dark_toggle_btn = QPushButton("🌙 다크")
            self.dark_toggle_btn.setFixedWidth(72)
            self.dark_toggle_btn.setToolTip("다크/라이트 모드 전환")
            self.dark_toggle_btn.clicked.connect(self._toggle_dark_mode)
            bottom = QHBoxLayout()
            bottom.addWidget(self.status_label)
            bottom.addWidget(self.progress, stretch=1)
            bottom.addWidget(self.eta_label)
            bottom.addWidget(self.dark_toggle_btn)
            layout.addLayout(bottom)
            self._ensure_default_llm_ready()

        def _build_controls(self) -> QWidget:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)

            io_group = QGroupBox("입력 설정")
            io_grid = QGridLayout(io_group)
            io_grid.setColumnMinimumWidth(0, 160)
            io_grid.setColumnStretch(0, 0)
            io_grid.setColumnStretch(1, 1)
            io_grid.setColumnStretch(2, 0)
            io_grid.setHorizontalSpacing(10)
            io_grid.setVerticalSpacing(10)
            self.input_edit = QLineEdit(str(Path.cwd() / "reports"))
            self.input_edit.setMinimumWidth(200)
            self.input_edit.setToolTip("폴더 1개 또는 PDF 여러 개를 선택할 수 있습니다.")
            self.metadata_edit = QLineEdit("")
            self.metadata_edit.setMinimumWidth(200)
            self.metadata_edit.setPlaceholderText("비워두면 내장 최신 metadata(default_metadata)를 사용합니다.")
            self.metadata_edit.setToolTip("비워두면 최신 내장 metadata를 사용합니다. 외부 JSON 파일을 지정할 수도 있습니다.")
            self.kb_edit = QLineEdit(str(Path.cwd() / "esgreveal_knowledge_base"))
            self.kb_edit.setMinimumWidth(200)
            self.output_edit = QLineEdit(str(Path.cwd() / "gui_output"))
            self.output_edit.setMinimumWidth(200)
            self.recursive_check = QCheckBox("하위 폴더까지 포함")
            self.recursive_check.setChecked(True)
            self.recursive_check.setToolTip("입력으로 폴더를 고르면 하위 폴더까지 함께 검색합니다.")
            self.use_kb_check = QCheckBox("추출 시 Knowledge Base 우선 사용")
            self.use_kb_check.setChecked(True)
            self.use_kb_check.setToolTip("추출할 때 미리 만든 Knowledge Base를 먼저 검색합니다.")
            self.resume_checkpoint_check = QCheckBox("Knowledge Base 체크포인트 재개 사용")
            self.resume_checkpoint_check.setChecked(True)
            self.resume_checkpoint_check.setToolTip("이전에 중간까지 만든 Knowledge Base가 있으면 이어서 다시 시작합니다.")

            io_grid.addWidget(QLabel("입력 PDF/폴더"), 0, 0)
            io_grid.addWidget(self.input_edit, 0, 1)
            browse_input = QPushButton("선택")
            browse_input.setFixedWidth(96)
            browse_input.clicked.connect(self._choose_input)
            io_grid.addWidget(browse_input, 0, 2)

            io_grid.addWidget(QLabel("메타데이터 (기본값: 내장 최신)"), 1, 0)
            io_grid.addWidget(self.metadata_edit, 1, 1)
            browse_meta = QPushButton("JSON 선택")
            browse_meta.setFixedWidth(96)
            browse_meta.clicked.connect(self._choose_metadata)
            io_grid.addWidget(browse_meta, 1, 2)

            io_grid.addWidget(QLabel("Knowledge Base 폴더"), 2, 0)
            io_grid.addWidget(self.kb_edit, 2, 1)
            browse_kb = QPushButton("찾기")
            browse_kb.setFixedWidth(96)
            browse_kb.clicked.connect(self._choose_kb_dir)
            io_grid.addWidget(browse_kb, 2, 2)

            io_grid.addWidget(QLabel("결과 폴더"), 3, 0)
            io_grid.addWidget(self.output_edit, 3, 1)
            browse_out = QPushButton("찾기")
            browse_out.setFixedWidth(96)
            browse_out.clicked.connect(self._choose_output_dir)
            io_grid.addWidget(browse_out, 3, 2)

            io_grid.addWidget(self.recursive_check, 4, 0, 1, 3)
            io_grid.addWidget(self.use_kb_check, 5, 0, 1, 3)
            io_grid.addWidget(self.resume_checkpoint_check, 6, 0, 1, 3)
            layout.addWidget(io_group)

            kb_group = QGroupBox("Knowledge Base 모드")
            kb_grid = QGridLayout(kb_group)
            self.kb_summary_mode_combo = QComboBox()
            self.kb_summary_mode_combo.addItem("논문형 mt5 요약 모드", "mt5")
            self.kb_summary_mode_combo.addItem("실험용 빠른 모드 (추출적 요약 + 원문 벡터)", "extractive")
            self.kb_summary_mode_combo.addItem("요약 끔 (원문 벡터만 사용)", "off")
            self.kb_summary_mode_combo.setCurrentIndex(0)
            kb_grid.addWidget(QLabel("요약 / 벡터 모드"), 0, 0)
            kb_grid.addWidget(self.kb_summary_mode_combo, 0, 1)
            layout.addWidget(kb_group)

            retrieval_group = QGroupBox("검색 설정")
            retrieval_grid = QGridLayout(retrieval_group)
            self.topk_spin = QSpinBox()
            self.topk_spin.setRange(1, 20)
            self.topk_spin.setValue(5)
            self.topk_spin.setToolTip(
                "각 KPI의 공시 여부와 수치를 검토할 때 최종적으로 참고할 후보 수입니다.\n"
                "값이 크면 더 넓게 확인하고, 값이 작으면 더 간단하게 판단합니다.\n"
                "권장: 3~7"
            )
            self.initialk_spin = QSpinBox()
            self.initialk_spin.setRange(1, 100)
            self.initialk_spin.setValue(20)
            self.initialk_spin.setToolTip(
                "각 KPI를 검토할 때 처음 넓게 살펴볼 후보 수입니다.\n"
                "값이 크면 놓칠 가능성은 줄고, 처리 시간은 조금 더 늘어납니다.\n"
                "권장: 15~30"
            )
            topk_label = QLabel("최종 Top-K")
            topk_label.setToolTip(self.topk_spin.toolTip())
            initialk_label = QLabel("초기 후보 수")
            initialk_label.setToolTip(self.initialk_spin.toolTip())
            retrieval_grid.addWidget(topk_label, 0, 0)
            retrieval_grid.addWidget(self.topk_spin, 0, 1)
            retrieval_grid.addWidget(initialk_label, 1, 0)
            retrieval_grid.addWidget(self.initialk_spin, 1, 1)
            layout.addWidget(retrieval_group)

            llm_group = QGroupBox("LLM 설정")
            llm_grid = QGridLayout(llm_group)
            llm_grid.setColumnMinimumWidth(0, 120)
            llm_grid.setColumnStretch(0, 0)
            llm_grid.setColumnStretch(1, 1)
            llm_grid.setHorizontalSpacing(10)
            llm_grid.setVerticalSpacing(8)
            self.llm_backend_combo = QComboBox()
            self.llm_backend_combo.addItem("휴리스틱만 사용", "none")
            self.llm_backend_combo.addItem("Codex CLI", "codex_cli")
            self.llm_backend_combo.addItem("Gemini CLI", "gemini_cli")
            self.llm_backend_combo.addItem("Claude Code", "claude_cli")
            self.llm_backend_combo.setCurrentIndex(max(0, self.llm_backend_combo.findData("gemini_cli")))
            self.llm_backend_combo.currentIndexChanged.connect(self._sync_llm_fields)
            self.llm_model_edit = QComboBox()
            self.llm_model_edit.setEditable(False)
            self.llm_model_edit.setMinimumContentsLength(24)
            self.llm_model_edit.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.llm_model_edit.setToolTip("Backend별 추천 모델 목록에서 선택합니다.")
            self.reasoning_effort_combo = QComboBox()
            self.reasoning_effort_combo.addItems(["medium", "high", "xhigh", "low", "max"])
            self.reasoning_effort_combo.setCurrentText("xhigh")
            self.llm_model_edit.currentTextChanged.connect(self._sync_reasoning_default)
            self.llm_timeout_spin = QSpinBox()
            self.llm_timeout_spin.setRange(30, 3600)
            self.llm_timeout_spin.setValue(180)
            self.codex_executable_edit = QLineEdit("codex")
            self.codex_workdir_edit = QLineEdit(str(Path.cwd()))
            self.gemini_executable_edit = QLineEdit("gemini.cmd")
            self.gemini_workdir_edit = QLineEdit(str(Path.cwd()))
            self.claude_executable_edit = QLineEdit("claude")
            self.claude_workdir_edit = QLineEdit(str(Path.cwd()))
            llm_grid.addWidget(QLabel("Backend"), 0, 0)
            llm_grid.addWidget(self.llm_backend_combo, 0, 1)
            llm_grid.addWidget(QLabel("Model"), 1, 0)
            llm_grid.addWidget(self.llm_model_edit, 1, 1)
            llm_grid.addWidget(QLabel("Reasoning"), 2, 0)
            llm_grid.addWidget(self.reasoning_effort_combo, 2, 1)
            llm_grid.addWidget(QLabel("Timeout"), 3, 0)
            llm_grid.addWidget(self.llm_timeout_spin, 3, 1)
            llm_grid.addWidget(QLabel("Codex CLI"), 4, 0)
            llm_grid.addWidget(self.codex_executable_edit, 4, 1)
            llm_grid.addWidget(QLabel("Codex Workdir"), 5, 0)
            llm_grid.addWidget(self.codex_workdir_edit, 5, 1)
            llm_grid.addWidget(QLabel("Gemini CLI"), 6, 0)
            llm_grid.addWidget(self.gemini_executable_edit, 6, 1)
            llm_grid.addWidget(QLabel("Gemini Workdir"), 7, 0)
            llm_grid.addWidget(self.gemini_workdir_edit, 7, 1)
            llm_grid.addWidget(QLabel("Claude Code"), 8, 0)
            llm_grid.addWidget(self.claude_executable_edit, 8, 1)
            llm_grid.addWidget(QLabel("Claude Workdir"), 9, 0)
            llm_grid.addWidget(self.claude_workdir_edit, 9, 1)
            layout.addWidget(llm_group)
            self._sync_llm_fields()

            buttons_group = QGroupBox("실행")
            buttons_layout = QVBoxLayout(buttons_group)
            self.build_kb_button = QPushButton("1. Knowledge Base 구축")
            self.build_kb_button.clicked.connect(self._on_build_kb)
            self.run_extract_button = QPushButton("2. KPI 추출")
            self.run_extract_button.clicked.connect(self._on_run_extract)
            self.save_json_button = QPushButton("현재 결과 JSON 저장")
            self.save_json_button.clicked.connect(self._save_current_json)
            self.save_md_button = QPushButton("현재 결과 Markdown 저장")
            self.save_md_button.clicked.connect(self._save_current_markdown)
            self.save_json_button.setEnabled(False)
            self.save_md_button.setEnabled(False)
            self._step_buttons = [
                self.build_kb_button,
                self.run_extract_button,
                self.save_json_button,
                self.save_md_button,
            ]
            buttons_layout.addWidget(self.build_kb_button)
            buttons_layout.addWidget(self.run_extract_button)
            buttons_layout.addWidget(self.save_json_button)
            buttons_layout.addWidget(self.save_md_button)
            layout.addWidget(buttons_group)
            self._highlight_step(self.build_kb_button)

            log_group = QGroupBox("로그")
            log_layout = QVBoxLayout(log_group)
            self.log_view = QPlainTextEdit()
            self.log_view.setReadOnly(True)
            log_layout.addWidget(self.log_view)
            layout.addWidget(log_group, stretch=1)
            return widget

        def _build_results(self) -> QWidget:
            widget = QWidget()
            layout = QVBoxLayout(widget)

            header = QHBoxLayout()
            header.addWidget(QLabel("문서"))
            self.run_selector = QComboBox()
            self.run_selector.currentIndexChanged.connect(self._on_run_changed)
            header.addWidget(self.run_selector, stretch=1)
            layout.addLayout(header)

            main_splitter = QSplitter(Qt.Horizontal)
            main_splitter.setChildrenCollapsible(False)
            layout.addWidget(main_splitter, stretch=1)

            self.kpi_list = QListWidget()
            self.kpi_list.currentRowChanged.connect(self._on_result_changed)
            main_splitter.addWidget(self.kpi_list)

            right = QTabWidget()

            # ── 문서 요약 ──────────────────────────────────────────
            self.summary_view = QPlainTextEdit()
            self.summary_view.setReadOnly(True)
            self.summary_view.setFont(_monospace_font())
            right.addTab(self.summary_view, "문서 요약")

            # ── 검색 근거 (테이블) ──────────────────────────────────
            self.context_table = QTableWidget(0, 5)
            self.context_table.setHorizontalHeaderLabels(["페이지", "종류", "점수", "섹션 / 아웃라인", "매치 키워드"])
            self.context_table.horizontalHeader().setStretchLastSection(True)
            self.context_table.setSelectionBehavior(QTableWidget.SelectRows)
            self.context_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.context_table.verticalHeader().setVisible(False)
            right.addTab(self.context_table, "검색 근거 (Top-K)")

            # ── 근거 원문 ──────────────────────────────────────────
            self.context_text = QPlainTextEdit()
            self.context_text.setReadOnly(True)
            self.context_text.setFont(_monospace_font())
            right.addTab(self.context_text, "근거 원문")

            # ── LLM Prompt ─────────────────────────────────────────
            self.prompt_view = QPlainTextEdit()
            self.prompt_view.setReadOnly(True)
            self.prompt_view.setFont(_monospace_font())
            right.addTab(self.prompt_view, "LLM Prompt")

            # ── 출력 결과 ──────────────────────────────────────────
            self.output_view = QPlainTextEdit()
            self.output_view.setReadOnly(True)
            self.output_view.setFont(_monospace_font())
            right.addTab(self.output_view, "출력 결과")

            # ── 마크다운 뷰 ────────────────────────────────────────
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView
                self.md_view: QWebEngineView | QPlainTextEdit = QWebEngineView()
            except Exception:
                self.md_view = QPlainTextEdit()
                self.md_view.setReadOnly(True)
                self.md_view.setFont(_monospace_font())
            right.addTab(self.md_view, "마크다운")

            # ── PDF 뷰 ─────────────────────────────────────────────
            self._pdf_view_widget, self._pdf_load_fn = _build_pdf_viewer()
            right.addTab(self._pdf_view_widget, "원본 PDF")

            main_splitter.addWidget(right)
            main_splitter.setSizes([300, 780])
            return widget

        def _choose_input(self) -> None:
            choice = QMessageBox.question(
                self,
                "입력 선택",
                "폴더 1개를 고르시겠습니까?\n아니오를 누르면 PDF 여러 개를 선택합니다.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if choice == QMessageBox.Cancel:
                return
            if choice == QMessageBox.Yes:
                path = QFileDialog.getExistingDirectory(self, "입력 폴더 선택", self.input_edit.text() or str(Path.cwd()))
                if path:
                    self.input_edit.setText(path)
                return

            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "입력 PDF 여러 개 선택",
                self.input_edit.text() or str(Path.cwd()),
                "PDF Files (*.pdf)",
            )
            if file_paths:
                self.input_edit.setText(_format_input_targets([Path(path) for path in file_paths]))

        def _choose_metadata(self) -> None:
            file_path, _ = QFileDialog.getOpenFileName(self, "메타데이터 선택", self.metadata_edit.text(), "JSON Files (*.json)")
            if file_path:
                self.metadata_edit.setText(file_path)

        def _choose_kb_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "Knowledge Base 폴더 선택", self.kb_edit.text())
            if path:
                self.kb_edit.setText(path)

        def _choose_output_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(self, "결과 폴더 선택", self.output_edit.text())
            if path:
                self.output_edit.setText(path)

        def _sync_llm_fields(self) -> None:
            backend = self.llm_backend_combo.currentData()
            is_codex = backend == "codex_cli"
            is_gemini = backend == "gemini_cli"
            is_claude = backend == "claude_cli"
            self.reasoning_effort_combo.setEnabled(is_codex or is_claude)
            self.codex_executable_edit.setEnabled(is_codex)
            self.codex_workdir_edit.setEnabled(is_codex)
            self.gemini_executable_edit.setEnabled(is_gemini)
            self.gemini_workdir_edit.setEnabled(is_gemini)
            self.claude_executable_edit.setEnabled(is_claude)
            self.claude_workdir_edit.setEnabled(is_claude)
            current_model = self.llm_model_edit.currentText().strip()
            self.llm_model_edit.blockSignals(True)
            self.llm_model_edit.clear()
            suggested_models = suggested_models_for_backend(str(backend))
            for model_name in suggested_models:
                self.llm_model_edit.addItem(model_name)
            default_model = ""
            if backend == "gemini_cli":
                default_model = "gemini-3-pro-preview"
                if default_model not in suggested_models and suggested_models:
                    default_model = suggested_models[0]
            elif suggested_models:
                default_model = suggested_models[0]
            if current_model and current_model in suggested_models:
                self.llm_model_edit.setCurrentText(current_model)
            elif default_model:
                self.llm_model_edit.setCurrentText(default_model)
            elif self.llm_model_edit.count() > 0:
                self.llm_model_edit.setCurrentIndex(0)
            self.llm_model_edit.blockSignals(False)
            self._sync_reasoning_default()

        def _ensure_default_llm_ready(self) -> None:
            backend = str(self.llm_backend_combo.currentData())
            if backend != "gemini_cli" or self._gemini_install_prompted:
                return
            self._gemini_install_prompted = True
            executable = self.gemini_executable_edit.text().strip() or "gemini.cmd"
            if cli_is_available(executable):
                return

            message = (
                "Gemini CLI가 설치되어 있지 않습니다.\n\n"
                "기본 LLM은 Gemini CLI이며, 개인 Google 계정 기준 무료 티어를 사용할 수 있습니다.\n"
                "Codex와 Claude Code는 토큰/비용 부담이 더 클 수 있습니다.\n\n"
                "지금 Gemini CLI 설치를 시도할까요?\n"
                "공식 설치 명령: npm install -g @google/gemini-cli"
            )
            choice = QMessageBox.question(self, "Gemini CLI 설치", message)
            if choice != QMessageBox.Yes:
                return

            npm_executable = shutil.which("npm.cmd") or shutil.which("npm")
            if not npm_executable:
                QMessageBox.information(
                    self,
                    "Gemini CLI 설치 불가",
                    "npm이 설치되어 있지 않아 자동 설치를 진행할 수 없습니다.\nNode.js 설치 후 다시 시도해 주세요.",
                )
                return

            self.status_label.setText("Gemini CLI 설치 중...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                _no_win = {"creationflags": 0x08000000} if __import__("os").name == "nt" else {}
                completed = subprocess.run(
                    [npm_executable, "install", "-g", "@google/gemini-cli@latest"],
                    cwd=str(Path.cwd()),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=900,
                    **_no_win,
                )
            finally:
                QApplication.restoreOverrideCursor()

            if completed.returncode == 0 and cli_is_available(executable):
                self.log_view.appendPlainText("Gemini CLI 설치 완료")
                QMessageBox.information(self, "Gemini CLI 설치", "Gemini CLI 설치가 완료되었습니다.")
                return

            self.log_view.appendPlainText(completed.stdout)
            self.log_view.appendPlainText(completed.stderr)
            QMessageBox.warning(
                self,
                "Gemini CLI 설치 실패",
                "자동 설치에 실패했습니다.\n"
                "Gemini CLI는 무료 티어로 쓸 수 있고, Codex/Claude보다 비용 부담이 적습니다.\n"
                "터미널에서 `npm install -g @google/gemini-cli@latest`를 직접 실행해 주세요.",
            )

        def _sync_reasoning_default(self) -> None:
            backend = str(self.llm_backend_combo.currentData())
            model = self.llm_model_edit.currentText().strip()
            self.reasoning_effort_combo.setCurrentText(default_reasoning_effort_for_model(backend, model))

        def _build_llm_config(self) -> LLMConfig | None:
            backend = self.llm_backend_combo.currentData()
            if backend == "none":
                return None
            if backend == "codex_cli":
                return LLMConfig(
                    backend="codex_cli",
                    model=self.llm_model_edit.currentText().strip() or "gpt-5.4-mini",
                    timeout=self.llm_timeout_spin.value(),
                    reasoning_effort=self.reasoning_effort_combo.currentText(),
                    codex_executable=self.codex_executable_edit.text().strip() or "codex",
                    codex_workdir=self.codex_workdir_edit.text().strip(),
                )
            if backend == "gemini_cli":
                return LLMConfig(
                    backend="gemini_cli",
                    model=self.llm_model_edit.currentText().strip(),
                    timeout=self.llm_timeout_spin.value(),
                    gemini_executable=self.gemini_executable_edit.text().strip() or "gemini.cmd",
                    gemini_workdir=self.gemini_workdir_edit.text().strip(),
                )
            if backend == "claude_cli":
                return LLMConfig(
                    backend="claude_cli",
                    model=self.llm_model_edit.currentText().strip(),
                    timeout=self.llm_timeout_spin.value(),
                    reasoning_effort=self.reasoning_effort_combo.currentText(),
                    claude_executable=self.claude_executable_edit.text().strip() or "claude",
                    claude_workdir=self.claude_workdir_edit.text().strip(),
                )
            raise ValueError(f"Unsupported LLM backend: {backend}")

        def _backend_label(self, backend: str) -> str:
            mapping = {
                "none": "휴리스틱만 사용",
                "codex_cli": "Codex CLI",
                "gemini_cli": "Gemini CLI",
                "claude_cli": "Claude Code",
            }
            return mapping.get(backend, backend)

        def _build_llm_config_for_backend(self, backend: str, model: str) -> LLMConfig | None:
            if backend == "none":
                return None
            if backend == "codex_cli":
                return LLMConfig(
                    backend="codex_cli",
                    model=model or "gpt-5.4-mini",
                    timeout=self.llm_timeout_spin.value(),
                    reasoning_effort=default_reasoning_effort_for_model("codex_cli", model),
                    codex_executable=self.codex_executable_edit.text().strip() or "codex",
                    codex_workdir=self.codex_workdir_edit.text().strip(),
                )
            if backend == "gemini_cli":
                return LLMConfig(
                    backend="gemini_cli",
                    model=model,
                    timeout=self.llm_timeout_spin.value(),
                    gemini_executable=self.gemini_executable_edit.text().strip() or "gemini.cmd",
                    gemini_workdir=self.gemini_workdir_edit.text().strip(),
                )
            if backend == "claude_cli":
                return LLMConfig(
                    backend="claude_cli",
                    model=model,
                    timeout=self.llm_timeout_spin.value(),
                    reasoning_effort=default_reasoning_effort_for_model("claude_cli", model),
                    claude_executable=self.claude_executable_edit.text().strip() or "claude",
                    claude_workdir=self.claude_workdir_edit.text().strip(),
                )
            raise ValueError(f"Unsupported LLM backend: {backend}")

        def _choose_alternate_llm(self, failed_backend: str) -> LLMConfig | None:
            backend_items = [
                (self._backend_label("gemini_cli"), "gemini_cli"),
                (self._backend_label("codex_cli"), "codex_cli"),
                (self._backend_label("claude_cli"), "claude_cli"),
            ]
            available: list[tuple[str, str]] = []
            for label, backend in backend_items:
                if backend == failed_backend:
                    continue
                if backend == "gemini_cli" and not cli_is_available(self.gemini_executable_edit.text().strip() or "gemini.cmd"):
                    continue
                if backend == "codex_cli" and not cli_is_available(self.codex_executable_edit.text().strip() or "codex"):
                    continue
                if backend == "claude_cli" and not cli_is_available(self.claude_executable_edit.text().strip() or "claude"):
                    continue
                available.append((label, backend))

            if not available:
                QMessageBox.information(self, "다른 LLM 없음", "현재 바로 시도할 수 있는 다른 LLM이 없습니다.")
                return None

            backend_label, ok = QInputDialog.getItem(
                self,
                "다른 LLM 선택",
                "다시 시도할 LLM을 선택하세요.",
                [label for label, _ in available],
                0,
                False,
            )
            if not ok:
                return None
            backend = next(code for label, code in available if label == backend_label)

            models = suggested_models_for_backend(backend)
            default_index = 0
            if backend == "gemini_cli":
                try:
                    default_index = models.index("gemini-3-pro-preview")
                except ValueError:
                    default_index = 0
            model, ok = QInputDialog.getItem(
                self,
                "모델 선택",
                "다시 시도할 모델을 선택하세요.",
                models,
                default_index,
                False,
            )
            if not ok:
                return None
            return self._build_llm_config_for_backend(backend, model)

        def _highlight_step(self, active_btn: Any) -> None:
            _ACTIVE = (
                "QPushButton {"
                "  background-color: #2563eb;"
                "  color: #ffffff;"
                "  border: 2px solid #1d4ed8;"
                "  border-radius: 5px;"
                "  padding: 5px 12px;"
                "  font-weight: bold;"
                "  border-bottom: 4px solid #1e40af;"  # 입체감
                "}"
                "QPushButton:hover { background-color: #1d4ed8; }"
                "QPushButton:pressed { border-bottom: 1px solid #1e40af; padding-top: 7px; }"
            )
            _IDLE = ""
            for btn in self._step_buttons:
                btn.setStyleSheet(_ACTIVE if btn is active_btn else _IDLE)

        def _set_busy(self, busy: bool, message: str = "") -> None:
            self.build_kb_button.setEnabled(not busy)
            self.run_extract_button.setEnabled(not busy)
            self.progress.setVisible(busy)
            self.eta_label.setVisible(busy)
            if busy:
                self.progress.setRange(0, 0)
                self.progress.setValue(0)
                self.eta_label.setText("")
            else:
                self.eta_label.hide()
            self.status_label.setText(message or ("작업 중..." if busy else "대기 중"))

        def _start_task(self, fn: Callable[..., Any], kwargs: dict[str, Any], done: Callable[[Any], None]) -> None:
            self._task_start_time = time.monotonic()
            self._thread = TaskThread(fn, kwargs)
            self._thread.progress.connect(self._handle_progress)
            self._thread.finished_ok.connect(done)
            self._thread.finished_ok.connect(lambda _: self._set_busy(False, "작업 완료"))
            self._thread.failed.connect(self._handle_failure)
            self._set_busy(True, "작업 중...")
            self._thread.start()

        def _fmt_seconds(self, seconds: float) -> str:
            seconds = int(seconds)
            if seconds < 60:
                return f"{seconds}초"
            m, s = divmod(seconds, 60)
            return f"{m}분 {s:02d}초"

        def _handle_progress(self, payload: dict[str, Any]) -> None:
            if payload.get("kind") == "log":
                msg = str(payload.get("message", ""))
                self.log_view.appendPlainText(msg)
                self._live_update_from_log(msg)
            elif payload.get("kind") == "progress":
                value = payload.get("value")
                total = payload.get("total", 100)
                message = payload.get("message", "")
                if value is None:
                    if message:
                        self.log_view.appendPlainText(message)
                        self._live_update_from_log(message)
                else:
                    self.progress.setRange(0, total)
                    self.progress.setValue(value)
                    pct = int(value * 100 / total) if total else 0
                    self.status_label.setText(f"{message} ({pct}%)" if message else f"작업 중... {pct}%")
                    elapsed = time.monotonic() - self._task_start_time
                    if value > 0 and total > 0 and elapsed > 2:
                        eta = elapsed / value * (total - value)
                        self.eta_label.setText(f"남은 시간: {self._fmt_seconds(eta)}")
                    elif value == 0:
                        self.eta_label.setText("")

        def _live_update_from_log(self, msg: str) -> None:
            """로그 메시지에서 KPI ID·결과를 파싱해 오른쪽 패널을 실시간 갱신한다."""
            import re as _re
            # "[n/N] filename 추출 중..." → run_selector 상태 표시
            m = _re.search(r"\[(\d+)/(\d+)\]\s+(.+?)\s+추출 중", msg)
            if m:
                idx_text = f"📄 {m.group(3)}  ({m.group(1)}/{m.group(2)})"
                # run_selector에 placeholder 아이템 추가 (완료 시 덮어쓰임)
                if self.run_selector.findText(idx_text) < 0:
                    self.run_selector.blockSignals(True)
                    # 현재 진행 중인 문서는 첫 번째 슬롯으로 유지
                    self.run_selector.addItem(idx_text)
                    self.run_selector.blockSignals(False)
                return

            # "KPI id: XXX → disclosure=1, value=..." 형태의 중간 결과가 있으면 output_view에 append
            if "disclosure" in msg.lower() or "kpi" in msg.lower():
                current_text = self.output_view.toPlainText()
                lines = current_text.splitlines() if current_text else []
                # 최근 30줄만 유지해 너무 길어지지 않게
                lines.append(msg)
                if len(lines) > 30:
                    lines = lines[-30:]
                self.output_view.setPlainText("\n".join(lines))
                self.output_view.verticalScrollBar().setValue(
                    self.output_view.verticalScrollBar().maximum()
                )
                return

            # 일반 진행 메시지는 summary_view에 append (작업 중 로그 미러)
            current = self.summary_view.toPlainText()
            lines = current.splitlines() if current else []
            lines.append(msg)
            if len(lines) > 60:
                lines = lines[-60:]
            self.summary_view.setPlainText("\n".join(lines))
            self.summary_view.verticalScrollBar().setValue(
                self.summary_view.verticalScrollBar().maximum()
            )

        def _toggle_dark_mode(self) -> None:
            self._dark_mode = not self._dark_mode
            if self._dark_mode:
                self.dark_toggle_btn.setText("☀️ 라이트")
                QApplication.instance().setStyleSheet(
                    "QWidget { background-color: #1e1e2e; color: #cdd6f4; }"
                    "QGroupBox { border: 1px solid #45475a; border-radius: 4px; margin-top: 6px; padding-top: 10px; }"
                    "QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #89b4fa; }"
                    "QLineEdit, QPlainTextEdit, QComboBox, QSpinBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 3px; padding: 2px 4px; }"
                    "QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4; selection-background-color: #45475a; }"
                    "QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 4px 10px; }"
                    "QPushButton:hover { background-color: #45475a; }"
                    "QPushButton:disabled { color: #585b70; }"
                    "QProgressBar { background-color: #313244; border: 1px solid #45475a; border-radius: 3px; text-align: center; color: #cdd6f4; }"
                    "QProgressBar::chunk { background-color: #89b4fa; border-radius: 2px; }"
                    "QTabWidget::pane { border: 1px solid #45475a; }"
                    "QTabBar::tab { background-color: #313244; color: #cdd6f4; padding: 4px 12px; border: 1px solid #45475a; }"
                    "QTabBar::tab:selected { background-color: #45475a; color: #89b4fa; }"
                    "QTableWidget { background-color: #1e1e2e; color: #cdd6f4; gridline-color: #45475a; }"
                    "QHeaderView::section { background-color: #313244; color: #89b4fa; border: 1px solid #45475a; padding: 3px; }"
                    "QListWidget { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }"
                    "QListWidget::item:selected { background-color: #45475a; color: #89b4fa; }"
                    "QScrollBar:vertical { background-color: #1e1e2e; width: 10px; }"
                    "QScrollBar::handle:vertical { background-color: #45475a; border-radius: 5px; min-height: 20px; }"
                    "QSplitter::handle { background-color: #45475a; }"
                    "QCheckBox { color: #cdd6f4; }"
                    "QLabel { color: #cdd6f4; }"
                )
            else:
                self.dark_toggle_btn.setText("🌙 다크")
                QApplication.instance().setStyleSheet("")

        def _retry_with_llm_config(self, llm_config: LLMConfig | None) -> None:
            if not self._last_extract_kwargs:
                return
            kwargs = dict(self._last_extract_kwargs)
            kwargs["llm_config"] = llm_config
            self._last_extract_kwargs = dict(kwargs)
            self._start_task(run_extract_task, kwargs, self._on_extract_done)

        def _handle_failure(self, payload: Any) -> None:
            self._set_busy(False, "작업 실패")
            if isinstance(payload, dict):
                trace = str(payload.get("traceback", ""))
                exc = payload.get("exception")
            else:
                trace = str(payload)
                exc = None

            if trace:
                self.log_view.appendPlainText(trace)

            if isinstance(exc, PipelineLLMError):
                backend_label = self._backend_label(exc.backend)
                failed_message = (
                    f"선택한 LLM이 실패했습니다.\n\n"
                    f"- KPI: {exc.kpi.kpi_ko}\n"
                    f"- LLM: {backend_label}\n"
                    f"- 모델: {exc.model or '-'}\n\n"
                    f"토큰/쿼터/실행 문제일 수 있습니다.\n"
                    f"다른 LLM으로 시도할까요?"
                )
                choice = QMessageBox.question(
                    self,
                    "LLM 실패",
                    failed_message,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if choice == QMessageBox.Yes:
                    alt_config = self._choose_alternate_llm(exc.backend)
                    if alt_config is not None:
                        self._retry_with_llm_config(alt_config)
                    return

                heuristic_choice = QMessageBox.question(
                    self,
                    "LLM 사용 불가",
                    "현재는 추출을 계속하지 않습니다.\n\nLLM 없이 규칙 기반 결과만 보시겠습니까?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if heuristic_choice == QMessageBox.Yes:
                    self._retry_with_llm_config(None)
                    return

                self.status_label.setText("LLM 실패로 추출이 중단되었습니다.")
                QMessageBox.warning(
                    self,
                    "추출 중단",
                    "LLM 사용이 불가하여 현재 추출을 중단했습니다.",
                )
                return

            QMessageBox.critical(self, "실패", trace or str(exc) or "알 수 없는 오류가 발생했습니다.")

        def _on_build_kb(self) -> None:
            input_targets = _parse_input_targets(self.input_edit.text())
            kb_dir = Path(self.kb_edit.text().strip())
            metadata = Path(self.metadata_edit.text().strip()) if self.metadata_edit.text().strip() else None
            if not input_targets or any(not path.exists() for path in input_targets):
                QMessageBox.warning(self, "입력 오류", "입력 경로가 없습니다.")
                return
            input_value: Any = input_targets[0] if len(input_targets) == 1 else input_targets
            self._start_task(
                build_kb_task,
                {
                    "input_path": input_value,
                    "output_dir": kb_dir,
                    "recursive": self.recursive_check.isChecked(),
                    "summary_model": None,
                    "embedding_model": None,
                    "summary_mode": str(self.kb_summary_mode_combo.currentData()),
                    "use_faiss": True,
                    "milvus_uri": None,
                    "milvus_collection": None,
                    "resume_checkpoints": self.resume_checkpoint_check.isChecked(),
                },
                self._on_kb_done,
            )

        def _on_kb_done(self, build: KnowledgeBaseBuild) -> None:
            self.log_view.appendPlainText(knowledge_base_to_json(build))
            self.status_label.setText(f"Knowledge Base 구축 완료: {build.record_count} records")
            self._highlight_step(self.run_extract_button)
            self._start_auto_extract_countdown(seconds=10)

        def _start_auto_extract_countdown(self, seconds: int = 30) -> None:
            from PySide6.QtCore import QTimer
            self._auto_extract_remaining = seconds
            self._auto_extract_timer = QTimer(self)
            self._auto_extract_timer.setInterval(1000)
            self._auto_extract_timer.timeout.connect(self._auto_extract_tick)
            self._auto_extract_timer.start()
            self.run_extract_button.clicked.connect(self._cancel_auto_extract)
            self._update_auto_extract_label()

        def _update_auto_extract_label(self) -> None:
            self.run_extract_button.setText(f"2. KPI 추출  (자동 시작 {self._auto_extract_remaining}초)")

        def _auto_extract_tick(self) -> None:
            self._auto_extract_remaining -= 1
            if self._auto_extract_remaining <= 0:
                self._cancel_auto_extract()
                self._on_run_extract()
            else:
                self._update_auto_extract_label()

        def _cancel_auto_extract(self) -> None:
            if hasattr(self, "_auto_extract_timer") and self._auto_extract_timer is not None:
                self._auto_extract_timer.stop()
                self._auto_extract_timer = None
            self.run_extract_button.setText("2. KPI 추출")
            try:
                self.run_extract_button.clicked.disconnect(self._cancel_auto_extract)
            except Exception:
                pass

        def _on_run_extract(self) -> None:
            self._cancel_auto_extract()
            input_targets = _parse_input_targets(self.input_edit.text())
            if not input_targets or any(not path.exists() for path in input_targets):
                QMessageBox.warning(self, "입력 오류", "입력 경로가 없습니다.")
                return
            metadata = Path(self.metadata_edit.text().strip()) if self.metadata_edit.text().strip() else None
            kb_dir = Path(self.kb_edit.text().strip()) if self.use_kb_check.isChecked() and self.kb_edit.text().strip() else None
            input_value: Any = input_targets[0] if len(input_targets) == 1 else input_targets
            kwargs = {
                "input_path": input_value,
                "metadata_path": metadata,
                "top_k": self.topk_spin.value(),
                "recursive": self.recursive_check.isChecked(),
                "llm_config": self._build_llm_config(),
                "kb_dir": kb_dir,
                "initial_k": self.initialk_spin.value(),
            }
            self._last_extract_kwargs = dict(kwargs)
            self._start_task(run_extract_task, kwargs, self._on_extract_done)

        def _on_extract_done(self, runs: list[PipelineRun]) -> None:
            self._runs = runs
            self.run_selector.blockSignals(True)
            self.run_selector.clear()
            for run in runs:
                self.run_selector.addItem(Path(run.source_path).name, run)
            self.run_selector.blockSignals(False)
            self.save_json_button.setEnabled(True)
            self.save_md_button.setEnabled(True)
            self._highlight_step(self.save_json_button)
            if runs:
                self.run_selector.setCurrentIndex(0)
                self._show_run(runs[0])

        def _on_run_changed(self, index: int) -> None:
            if index < 0 or index >= len(self._runs):
                return
            self._show_run(self._runs[index])

        def _show_run(self, run: PipelineRun) -> None:
            self._current_run = run
            s = run.document_summary
            kind_label = {"academic_paper": "논문/연구자료", "esg_report": "ESG 보고서"}.get(
                str(s.get("document_kind", "")), "미확인"
            )
            lines = [
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  문서 유형    {kind_label}",
                f"  추출 모드    {s.get('extraction_mode', '-')}",
                f"  페이지 수    {s.get('page_count', '-')}  (OCR: {s.get('ocr_page_count', 0)}p)",
                f"  블록 수      {s.get('block_count', '-')}  (표: {s.get('table_count', 0)}, 제목: {s.get('heading_count', 0)})",
                f"  섹션 수      {s.get('section_count', '-')}",
                f"  Retrieval    {s.get('retrieval_backend', '-')}",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            ]
            warnings = s.get("warnings") or []
            if warnings:
                lines.append("  ⚠ 경고")
                for w in warnings:
                    lines.append(f"    · {w}")
                lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            self.summary_view.setPlainText("\n".join(lines))
            self._refresh_md_view(run)
            self._pdf_load_fn(run.source_path)
            self.kpi_list.clear()
            for result in run.results:
                out = result.output if isinstance(result.output, dict) else {}
                disc = out.get("disclosure") or out.get("disclosed")
                unit_inferred = bool(out.get("UnitInferred") or out.get("unit_inferred"))
                dtype = result.metadata.get("disclosure_type", "")
                value = out.get("value") or out.get("Value")
                action = out.get("action") or out.get("Action")
                target = out.get("target") or out.get("Target")

                if disc is True or disc == 1:
                    if unit_inferred:
                        icon = "🔍"  # 공시됨, 단위 원문 확인 필요
                    elif dtype == "numeric" and not value:
                        icon = "⚠️"  # 공시는 됐으나 수치 미추출
                    else:
                        icon = "✅"
                elif disc is False or disc == 0:
                    icon = "❌"  # 명시적 미공시
                else:
                    # disclosed null — KPI 유형별로 판단
                    if dtype == "numeric" and value:
                        icon = "✅"  # 수치 있으면 사실상 공시
                    elif dtype != "numeric" and (action or target):
                        icon = "✅"  # 텍스트 KPI는 action/target이 있으면 공시
                    else:
                        icon = "❓"  # 판단 불가
                label = f"{icon} {result.metadata.get('id', '')}  {result.metadata.get('kpi_ko', '')}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, result)
                self.kpi_list.addItem(item)
            if run.results:
                self.kpi_list.setCurrentRow(0)

        def _refresh_md_view(self, run: PipelineRun) -> None:
            md_text = results_to_markdown(run)
            try:
                from PySide6.QtWebEngineWidgets import QWebEngineView
                if isinstance(self.md_view, QWebEngineView):
                    try:
                        import markdown as _md_lib
                        body = _md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
                    except Exception:
                        body = "<pre>" + md_text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"
                    html = (
                        "<html><head><meta charset='utf-8'>"
                        "<style>body{font-family:sans-serif;font-size:13px;padding:16px;line-height:1.6}"
                        "h1,h2{border-bottom:1px solid #ccc;padding-bottom:4px}"
                        "table{border-collapse:collapse;width:100%}"
                        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
                        "th{background:#f0f0f0}tr:nth-child(even){background:#f8f8f8}</style></head>"
                        f"<body>{body}</body></html>"
                    )
                    self.md_view.setHtml(html)
                    return
            except Exception:
                pass
            self.md_view.setPlainText(md_text)

        def _on_result_changed(self, row: int) -> None:
            if row < 0 or row >= self.kpi_list.count():
                return
            item = self.kpi_list.item(row)
            result = item.data(Qt.UserRole)
            self._show_result(result)

        def _show_result(self, result: Any) -> None:
            # ── 출력 결과 ──────────────────────────────────────────
            out = result.output if isinstance(result.output, dict) else {}
            disc = out.get("disclosure") or out.get("disclosed")
            unit_inferred = bool(out.get("UnitInferred") or out.get("unit_inferred"))
            dtype = result.metadata.get("disclosure_type", "")
            value = out.get("value") or out.get("Value")
            action = out.get("action") or out.get("Action")
            target = out.get("target") or out.get("Target")

            if disc is True or disc == 1:
                if unit_inferred:
                    disc_label = "🔍 공시됨  (단위 추론 — 원문 확인 필요)"
                elif dtype == "numeric" and not value:
                    disc_label = "⚠️ 공시됨  (수치 미추출 — 원문 확인 필요)"
                else:
                    disc_label = "✅ 공시됨"
            elif disc is False or disc == 0:
                disc_label = "❌ 미공시"
            else:
                if dtype == "numeric" and value:
                    disc_label = "✅ 공시됨  (수치 추출)"
                elif dtype != "numeric" and (action or target):
                    disc_label = "✅ 공시됨"
                else:
                    disc_label = "❓ 판단 불가"
            unit_str = (out.get('unit') or '')
            if unit_inferred and unit_str:
                unit_str += "  ⚠ 추론값"
            lines = [
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"  공시 여부    {disc_label}",
                f"  값           {out.get('value') or '-'}  {unit_str}".rstrip(),
                f"  목표         {out.get('target') or '-'}",
                f"  조치         {out.get('action') or '-'}",
                f"  신뢰도       {out.get('confidence') or '-'}",
                f"  출처 페이지  {out.get('source_page') or '-'}",
                "  ─────────────────────────────────────",
                f"  근거         {str(out.get('evidence') or '-')[:300]}",
            ]
            warnings = out.get("warnings") or []
            if warnings:
                lines.append("  ─────────────────────────────────────")
                for w in warnings:
                    lines.append(f"  ⚠ {w}")
            analysis = out.get("Analysis") or out.get("analysis")
            if analysis:
                lines += ["  ─────────────────────────────────────", f"  분석   {analysis}"]
            lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            lines.append("")
            lines.append("▼ raw JSON")
            lines.append(json.dumps(out, ensure_ascii=False, indent=2))
            self.output_view.setPlainText("\n".join(lines))

            # ── 근거 원문 ──────────────────────────────────────────
            ctx_lines: list[str] = []
            for index, entry in enumerate(result.retrieved_context):
                page = f"p.{entry.get('page_start')}-{entry.get('page_end')}"
                kind = entry.get("kind", "")
                score = entry.get("score", 0)
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                section = entry.get("section") or ""
                outline = " > ".join(entry.get("outline_path") or [])
                loc = outline or section or "—"
                ctx_lines.append(f"┌─ [{index + 1}]  {page}  │  {kind}  │  점수 {score_str}")
                ctx_lines.append(f"│  {loc}")
                ctx_lines.append("│")
                for text_line in (entry.get("text") or "").splitlines():
                    ctx_lines.append(f"│  {text_line}")
                ctx_lines.append("└" + "─" * 60)
                ctx_lines.append("")
            self.context_text.setPlainText("\n".join(ctx_lines))

            # ── 검색 근거 (테이블) ──────────────────────────────────
            ctx = result.retrieved_context
            self.context_table.setRowCount(len(ctx))
            for row, entry in enumerate(ctx):
                score = entry.get("score", 0)
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                section = entry.get("section") or ""
                outline = " > ".join(entry.get("outline_path") or [])
                loc = outline or section or "—"
                values = [
                    f"p.{entry.get('page_start')}-{entry.get('page_end')}",
                    str(entry.get("kind", "")),
                    score_str,
                    loc,
                    ", ".join(entry.get("matched_terms") or []),
                ]
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignVCenter | (Qt.AlignCenter if col < 3 else Qt.AlignLeft))
                    if col == 2:  # 점수 열 색상
                        try:
                            from PySide6.QtGui import QColor
                            sv = float(score_str)
                            if sv >= 0.7:
                                item.setForeground(QColor("#2d8a4e"))
                            elif sv >= 0.4:
                                item.setForeground(QColor("#b07d00"))
                            else:
                                item.setForeground(QColor("#c0392b"))
                        except Exception:
                            pass
                    self.context_table.setItem(row, col, item)
            self.context_table.resizeColumnToContents(0)
            self.context_table.resizeColumnToContents(1)
            self.context_table.resizeColumnToContents(2)

            # ── LLM Prompt ─────────────────────────────────────────
            self.prompt_view.setPlainText(_format_prompt_readable(result.prompt))

        def _save_current_json(self) -> None:
            if self._current_run is None:
                return
            output_dir = Path(self.output_edit.text().strip())
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{Path(self._current_run.source_path).stem}.json"
            path.write_text(results_to_json(self._current_run), encoding="utf-8")
            self.log_view.appendPlainText(f"저장: {path}")

        def _save_current_markdown(self) -> None:
            if self._current_run is None:
                return
            output_dir = Path(self.output_edit.text().strip())
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{Path(self._current_run.source_path).stem}.md"
            path.write_text(results_to_markdown(self._current_run), encoding="utf-8")
            self.log_view.appendPlainText(f"저장: {path}")

    app = QApplication.instance() or QApplication([])
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec()
