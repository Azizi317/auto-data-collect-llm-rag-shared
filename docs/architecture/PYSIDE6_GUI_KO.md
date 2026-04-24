# PySide6 GUI

이 프로젝트는 `PySide6` 기반 데스크톱 워크벤치를 포함한다.

구현 위치:

- [gui.py](../../esgreveal/gui.py)

## 가능한 작업

- PDF 또는 폴더 선택
- knowledge base 구축
- KPI 추출 실행
- KPI별 검색 근거 확인
- KPI별 LLM prompt 확인
- KPI별 출력 JSON 확인
- 현재 문서 결과를 JSON/Markdown으로 저장

## 실행

가장 쉬운 방법:

```powershell
.\ESGReveal GUI.cmd
```

직접 실행:

```powershell
.\ESGReveal Setup.cmd
.\.venv\Scripts\Activate.ps1
esgreveal-gui
```

또는 모듈 직접 실행:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "from esgreveal.gui import launch_gui; raise SystemExit(launch_gui())"
```

## 화면 구성

- 좌측: 입력 경로, metadata, ????? 폴더, 결과 폴더, 실행 버튼, 로그
- 우측 상단: 현재 문서 선택
- 우측 중앙: KPI 목록
- 우측 탭:
  - 문서 요약
  - 검색 근거
  - 근거 원문
  - LLM Prompt
  - 출력 JSON

## 비고

- `????? 폴더`가 준비되어 있고 `????? retrieval 우선 사용`이 켜져 있으면 knowledge vector base 검색을 먼저 사용한다.
- ?????가 없거나 실패하면 chunk fallback 검색으로 내려간다.
- 긴 작업은 UI가 멈추지 않도록 background thread에서 실행한다.
