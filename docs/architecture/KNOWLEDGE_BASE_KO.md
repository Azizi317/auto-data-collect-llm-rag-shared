# Structured Knowledge Base

이 프로젝트는 논문 흐름을 따라 ESG 보고서를 아래 세 구성요소로 정리한다.

- `textual contents`
- `document outlines`
- `table contents`

핵심 구현은 [knowledge_base.py](../../esgreveal/knowledge_base.py)에 있다.

## 1. Textual Contents

- [document.py](../../esgreveal/document.py)에서 문서 블록을 추출한다.
- [chunking.py](../../esgreveal/chunking.py)에서 문단 중심 청크를 만든다.
- 표 블록은 별도로 다루고, 일반 텍스트 청크를 `textual_contents` record로 저장한다.
- 각 record에는 `section`, `section_level`, `outline_path`, `source_blocks` 같은 추적 정보가 포함된다.

## 2. Document Outlines

- 제목 블록을 기준으로 outline record를 만든다.
- `outline_path`를 보존해서 retrieval 시 문서 구조 정보를 함께 활용한다.
- 각 outline record는 자신 아래에 속한 `textual_contents`와 `table_contents`를 연결할 수 있다.

## 3. Table Contents

- 표 블록은 `table_contents` record로 별도 저장한다.
- 표 본문 텍스트는 `Table Transformer / LORE-TSR / PyMuPDF fallback` 경로를 거쳐 복원한 결과를 우선 사용한다.
- 표 하나를 통째로 하나의 벡터로만 저장하지 않고 `one-to-many mapping`을 적용한다.
- 즉, `table_base` 1개와 `table_keyword` 여러 개로 분리한다.
- `table_keyword`는 주로 `indicator name / keyword / row label` 중심으로 만든다.
- 숫자만 있는 셀이나 부수 정보는 최대한 제외하고, KPI ontology와 다시 매칭해 score가 있는 keyword를 우선 남긴다.

## 4. 사용 모델

### 요약

- 기본 요약 모델: `google/mt5-small`
- 실패하면 extractive fallback으로 내려간다.

### 임베딩

- 기본 임베딩 모델: `dragonkue/BGE-m3-ko`
- 한국 ESG 보고서를 주 입력으로 삼기 때문에 한국어 중심 임베딩을 기본으로 둔다.

### Reranking

- 논문은 `coROM`으로 2차 reranking을 수행한다.
- 현재 구현은 같은 2단계 retrieval 철학을 유지하되, 한국어 성능을 위해 기본 reranker를 `dragonkue/bge-reranker-v2-m3-ko`로 둔다.
- 따라서 현재 구현은 `coROM exact model`이라기보다, 한국어 지향의 learned reranker 대체 구현으로 보는 것이 맞다.

## 5. 저장 파일

?????를 만들면 보통 아래 파일들이 생성된다.

- `records.jsonl`
- `textual_contents.jsonl`
- `document_outlines.jsonl`
- `table_contents.jsonl`
- `text_vectors.npy`
- `text_vectors.faiss`
- `manifest.json`

Milvus URI를 주면 같은 벡터를 Milvus collection에도 적재할 수 있다.

## 6. 실행 예시

먼저 환경을 준비한다.

```powershell
cd "<프로젝트_폴더>"
.\ESGReveal Setup.cmd
.\.venv\Scripts\Activate.ps1
```

단일 PDF 기준:

```powershell
python -m esgreveal --build-kb --pdf "reports\2025_kr\Hyundai_Motor_Sustainability_Report_2025_KOR.pdf" --kb-dir kb_hyundai
```

폴더 기준:

```powershell
python -m esgreveal --build-kb --input "reports\2025_kr" --recursive --kb-dir kb_2025_kr
```

Milvus Lite 예시:

```powershell
python -m esgreveal --build-kb --input "reports\2025_kr" --recursive --kb-dir kb_2025_kr --milvus-uri ".\kb_2025_kr\milvus_demo.db" --milvus-collection esgreveal_knowledge_base
```
