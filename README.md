# ESGReveal 한국형 구현

한국 ESG 보고서를 대상으로, 논문형 흐름에 맞춰 다음 단계를 구현한 프로젝트다.

- 보고서 전처리
- 구조 인식과 표 구조 분석
- `textual contents / document outlines / table contents` 기반 knowledge base 구축
- ESG Metadata query vector화, cosine retrieval, reranking
- KPI별 프롬프트 생성과 구조화 추출
- PySide6 GUI 워크벤치

## 프로젝트 구조

루트에서 주로 보면 되는 폴더는 아래 정도다.

- `esgreveal`: 본체 코드
- `reports`: 실제 ESG 보고서 PDF
- `docs`: 구현 기록과 사용 문서
- `scripts`: 환경 설정과 모델 준비 스크립트
- `tests`: 테스트
- `references`: 논문, 부록, 분석 메모
- `external`: 외부 레포 복사본

실행 결과용 폴더(`outputs`, `gui_output`, `esgreveal_knowledge_base`)는 배포본에 기본 포함하지 않고, 실행할 때 자동으로 다시 생긴다.

자세한 구조는 [PROJECT_STRUCTURE_KO.md](docs/architecture/PROJECT_STRUCTURE_KO.md)를 보면 된다.

## 빠른 시작

압축을 풀어 다른 사람과 공유해도, 이제는 경로에 상관없이 실행할 수 있다. 가장 쉬운 방법은 루트의 [ESGReveal GUI.cmd](ESGReveal%20GUI.cmd)를 더블클릭하는 것이다. 처음 실행이면 같은 폴더 안에 `.venv`를 만들고 필요한 패키지를 설치한 뒤 GUI를 연다.

수동으로 준비하려면:

```powershell
cd "<프로젝트_폴더>"
.\ESGReveal Setup.cmd
.\.venv\Scripts\Activate.ps1
```

단일 PDF 추출:

```powershell
python -m esgreveal --pdf "reports\2025_kr\Samsung_Electronics_Sustainability_Report_2025_KOR.pdf" --out result.json --markdown-out result.md
```

폴더 단위 추출:

```powershell
python -m esgreveal --input "reports\2025_kr" --recursive --kb-dir kb_2025_kr --out-dir out_json --markdown-dir out_md
```

Knowledge base 구축:

```powershell
python -m esgreveal --build-kb --input "reports\2025_kr" --recursive --kb-dir kb_2025_kr
```

GUI 실행:

```powershell
.\ESGReveal GUI.cmd
```

예쁜 아이콘 바로가기가 필요하면, 압축을 푼 뒤 아래 스크립트로 현재 PC 기준 `.lnk`를 다시 만들면 된다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_gui_shortcut.ps1
```

## 문서

- 프로젝트 개요: [PROJECT_OVERVIEW_KO.md](docs/PROJECT_OVERVIEW_KO.md)
- 문서 인덱스: [DOCS_INDEX_KO.md](docs/DOCS_INDEX_KO.md)
- 구현 기록: [IMPLEMENTATION_LOG_KO.md](docs/experiments/IMPLEMENTATION_LOG_KO.md)
- 구조 안내: [PROJECT_STRUCTURE_KO.md](docs/architecture/PROJECT_STRUCTURE_KO.md)
- Knowledge base: [KNOWLEDGE_BASE_KO.md](docs/architecture/KNOWLEDGE_BASE_KO.md)
- GUI: [PYSIDE6_GUI_KO.md](docs/architecture/PYSIDE6_GUI_KO.md)
- 런타임 설정: [LAB_RUNTIME_SETUP_KO.md](docs/setup/LAB_RUNTIME_SETUP_KO.md)
- 다음 작업: [NEXT_STEPS_KO.md](docs/experiments/NEXT_STEPS_KO.md)

## 외부 참고

- GeoLayoutLM: [AlibabaResearch/AdvancedLiterateMachinery](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/DocumentUnderstanding/GeoLayoutLM/README.md)
- Table Transformer: [microsoft/table-transformer](https://github.com/microsoft/table-transformer)

## Reranker 메모

- 논문은 top initial matches를 `coROM`으로 다시 평가해 순위를 정제하는 구조를 사용한다.
- 현재 프로젝트는 같은 철학의 2단계 reranking 구조를 유지하되, 기본 모델은 `dragonkue/bge-reranker-v2-m3-ko`를 사용한다.
- 이유는 공개된 `coROM` 계열 기본 체크포인트가 중국어/영어 중심이고, 현재 프로젝트는 한국 ESG 보고서를 대상으로 하기 때문이다.
- 따라서 현재 기본 설정은 `coROM exact checkpoint`가 아니라, 한국어 지원과 한국어 성능을 우선한 learned reranker 대체 구현이다.
