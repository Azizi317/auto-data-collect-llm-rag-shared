# ESGReveal 프로젝트 개요

이 문서는 이 저장소를 처음 보는 사람이 "이 프로젝트가 무엇을 하고, 어떤 흐름으로 동작하며, 어디를 보면 되는지"를 빠르게 이해하도록 만든 상위 설명 문서다.

## 1. 한 줄 요약

ESGReveal은 한국어 ESG 보고서 PDF를 읽어 문서 구조를 해석하고, KPI별 근거 문맥을 검색한 뒤, 정량/정성 공시 정보를 구조화된 JSON과 Markdown으로 추출하는 프로젝트다. 필요하면 같은 입력으로 검색용 knowledge base도 함께 구축할 수 있다.

## 2. 이 프로젝트가 해결하려는 문제

ESG 보고서는 표, 본문, 섹션 제목, 부록이 섞여 있고 같은 KPI도 표현 방식이 제각각이다. 이 프로젝트는 아래 문제를 줄이는 데 초점이 있다.

- PDF에서 본문과 표를 함께 읽기 어렵다.
- KPI마다 찾을 단어와 단위가 다르다.
- 단순 키워드 검색만으로는 공시 여부와 수치를 안정적으로 판별하기 어렵다.
- 나중에 재검색하려면 문서 내용을 검색 가능한 knowledge base 형태로 다시 정리해야 한다.

즉, 이 저장소는 "한국 ESG 보고서 -> 구조화된 블록/청크 -> KPI별 검색 -> 추출 결과" 흐름을 일관된 파이프라인으로 묶은 구현체다.

## 3. 핵심 사용 시나리오

이 프로젝트는 크게 네 가지 방식으로 쓴다.

1. 단일 PDF에서 KPI 추출
2. 폴더 단위 배치 추출
3. 추출 전에 knowledge base 구축
4. GUI에서 PDF 확인, 검색 설정, 추출 결과 검토

CLI 진입점은 [`esgreveal/cli.py`](../esgreveal/cli.py)이고, GUI 진입점은 [`esgreveal/gui.py`](../esgreveal/gui.py)다.

## 4. 전체 동작 흐름

### 4-1. 문서 파싱

[`esgreveal/document.py`](../esgreveal/document.py)는 PDF를 읽어 `DocumentBlock` 목록으로 바꾼다.

- 블록 종류: `heading`, `paragraph`, `table`, `ocr`, `mixed`
- 함께 유지되는 정보: 페이지 번호, 순서, 섹션명, outline path, 표 인덱스, 추출 소스
- 요약 정보: 페이지 수, 블록 수, 표 수, 섹션 수, OCR 사용 여부, 문서 유형(`esg_report` / `academic_paper`)

이 단계는 이후 검색과 추출에 필요한 문서 구조를 만드는 기초 계층이다.

### 4-2. 청크 생성

[`esgreveal/chunking.py`](../esgreveal/chunking.py)는 블록을 `DocumentChunk`로 묶는다.

- 같은 섹션/문맥을 유지하면서 청크를 구성한다.
- 너무 긴 텍스트는 문장 경계 근처에서 나눈다.
- 표와 본문을 분리하거나 `mixed`로 관리해 retrieval 단계에서 활용한다.

즉, 모델이 검색할 최소 단위를 만드는 계층이다.

### 4-3. KPI 메타데이터 준비

[`esgreveal/catalog.py`](../esgreveal/catalog.py)와 [`esgreveal/entities.py`](../esgreveal/entities.py)는 KPI 정의를 제공한다.

- 환경 / 사회 / 지배구조 카테고리
- 한국어/영문 KPI 이름
- 토픽, 단위, 공시 유형(`numeric`, `text`)
- 표준 참조와 K-ESG crosswalk
- 검색용 동의어와 지식 단서
- 출력 형식을 결정하는 expression template

프로젝트 기본 메타데이터를 그대로 쓸 수도 있고, 외부 JSON을 읽어 덮어쓸 수도 있다.

### 4-4. 검색과 랭킹

검색은 두 계층 중 하나로 동작한다.

#### A. chunk fallback

knowledge base가 없으면 [`esgreveal/retrieval.py`](../esgreveal/retrieval.py)가 현재 PDF에서 만든 청크를 직접 점수화한다.

- 용어 토큰 매칭
- 구절 포함 여부
- 섹션/outline 보정
- 표/본문 유형 가중치
- semantic similarity

#### B. knowledge vector base

`records.jsonl`과 `text_vectors.npy`가 있으면 [`esgreveal/agent_retrieval.py`](../esgreveal/agent_retrieval.py)가 knowledge base를 우선 사용한다.

- KPI를 질의 벡터로 변환
- cosine score로 초기 후보를 고른다.
- 규칙 기반 rerank를 한 번 더 적용한다.
- 최종적으로 `coROM` 계열 철학을 따르는 재정렬 단계를 거쳐 상위 문맥을 반환한다.

파이프라인은 KB가 준비되어 있으면 `knowledge_vector_base`, 아니면 `chunk_fallback`으로 동작한다.

### 4-5. 추출

[`esgreveal/pipeline.py`](../esgreveal/pipeline.py)는 KPI마다 아래 작업을 반복한다.

1. KPI 메타데이터 로드
2. 상위 문맥 retrieval
3. 프롬프트 생성
4. LLM 또는 heuristic 방식으로 결과 생성
5. 결과를 표준 출력 스키마로 정규화

LLM을 쓰지 않으면 규칙 기반 fallback을 사용한다. LLM을 쓰면 backend별 설정은 [`esgreveal/llm.py`](../esgreveal/llm.py)에서 관리한다.

지원 backend:

- `codex_cli`
- `gemini_cli`
- `claude_cli`
- 또는 `none`(heuristic only)

### 4-6. 산출물 저장과 검토

추출 결과는 두 형식으로 저장된다.

- JSON: 프로그램 후처리용
- Markdown: 사람이 검토하기 쉬운 리포트용

knowledge base를 만들 때는 아래 파일이 생성된다.

- `records.jsonl`
- `textual_contents.jsonl`
- `document_outlines.jsonl`
- `table_contents.jsonl`
- `text_vectors.npy`
- 선택적으로 `text_vectors.faiss`
- `manifest.json`

GUI에서는 추출 결과, 프롬프트, 근거 문맥, PDF 뷰어를 한 화면에서 함께 본다.

## 5. 주요 모듈 지도

| 경로 | 역할 |
| --- | --- |
| [`esgreveal/cli.py`](../esgreveal/cli.py) | 단일/배치 추출, knowledge base 구축용 CLI |
| [`esgreveal/gui.py`](../esgreveal/gui.py) | PySide6 기반 워크벤치 GUI |
| [`esgreveal/document.py`](../esgreveal/document.py) | PDF 블록 추출과 문서 요약 |
| [`esgreveal/chunking.py`](../esgreveal/chunking.py) | 블록을 검색용 청크로 변환 |
| [`esgreveal/retrieval.py`](../esgreveal/retrieval.py) | 청크 기반 fallback retrieval |
| [`esgreveal/agent_retrieval.py`](../esgreveal/agent_retrieval.py) | knowledge base 기반 retrieval와 reranking |
| [`esgreveal/knowledge_base.py`](../esgreveal/knowledge_base.py) | 요약, 임베딩, 인덱스, checkpoint를 포함한 KB 구축 |
| [`esgreveal/catalog.py`](../esgreveal/catalog.py) | 기본 KPI 메타데이터 정의 |
| [`esgreveal/llm.py`](../esgreveal/llm.py) | LLM backend 설정과 호출 정규화 |
| [`tests`](../tests) | 파이프라인, KB, retrieval, 출력 형식 검증 |

## 6. 폴더 구조를 어떻게 보면 되는가

루트에서 먼저 보면 되는 경로는 아래다.

- [`esgreveal`](../esgreveal): 본체 코드
- [`reports`](../reports): 실제 ESG 보고서 PDF 입력
- [`docs`](./): 설계 문서와 실험 메모
- [`scripts`](../scripts): 환경 설정, 모델 준비, 바로가기 생성
- [`tests`](../tests): 회귀 테스트
- [`references`](../references): 논문과 보조 자료
- [`external`](../external): 외부 레포 사본

실행 중 생기는 대표 산출물 폴더:

- `gui_output`: GUI 저장 결과
- `esgreveal_knowledge_base`: knowledge base 결과
- 기타 사용자 지정 output 폴더

## 7. 일반적인 작업 순서

처음 쓰는 사용자의 표준 흐름은 아래에 가깝다.

1. `reports`에 ESG PDF를 넣는다.
2. 필요하면 knowledge base를 먼저 만든다.
3. 단일 PDF 또는 폴더 단위로 KPI를 추출한다.
4. JSON/Markdown 결과를 저장한다.
5. GUI에서 PDF와 근거 문맥을 함께 검토한다.

예시 명령:

```powershell
python -m esgreveal --build-kb --input "reports\2025_kr" --recursive --kb-dir kb_2025_kr
python -m esgreveal --input "reports\2025_kr" --recursive --kb-dir kb_2025_kr --out-dir out_json --markdown-dir out_md
.\ESGReveal GUI.cmd
```

## 8. 이 프로젝트를 읽을 때 알아두면 좋은 점

- 한국어 ESG 보고서가 주 타깃이다.
- LLM은 선택 사항이고, 없으면 heuristic fallback이 동작한다.
- GUI는 PySide6가 필요하다.
- knowledge base는 요약, 임베딩, FAISS/Milvus 연동까지 포함한 별도 자산이다.
- 외부 레포는 참고 또는 어댑터 연결용이며, 실제 사용자 진입점은 `esgreveal` 패키지다.

## 9. 먼저 읽으면 좋은 문서

- 문서 전체 인덱스: [DOCS_INDEX_KO.md](DOCS_INDEX_KO.md)
- 폴더 구조 중심 설명: [architecture/PROJECT_STRUCTURE_KO.md](architecture/PROJECT_STRUCTURE_KO.md)
- knowledge base 상세: [architecture/KNOWLEDGE_BASE_KO.md](architecture/KNOWLEDGE_BASE_KO.md)
- GUI 상세: [architecture/PYSIDE6_GUI_KO.md](architecture/PYSIDE6_GUI_KO.md)
- 실행 환경: [setup/LAB_RUNTIME_SETUP_KO.md](setup/LAB_RUNTIME_SETUP_KO.md)
- 구현 기록: [experiments/IMPLEMENTATION_LOG_KO.md](experiments/IMPLEMENTATION_LOG_KO.md)

## 10. 요약

이 저장소는 단순 PDF 파서가 아니라, 한국 ESG 보고서를 대상으로 한 "문서 구조 해석 + KPI retrieval + 구조화 추출 + knowledge base 구축 + GUI 검토"까지 포함한 end-to-end 워크벤치다. 새로운 기여자나 사용자라면 이 문서를 먼저 읽고, 이후 세부 설계 문서로 내려가면 흐름을 훨씬 빠르게 잡을 수 있다.
