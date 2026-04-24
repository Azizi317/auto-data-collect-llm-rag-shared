# ESGReveal 작업 기록

이 문서는 이번 구현에서 무엇을 어떻게 붙였는지 빠르게 다시 볼 수 있도록 요약한 기록이다.

## 1. 메타데이터 구조 정리

- 논문 용어에 맞춰 `Aspect / KPI / Topic / Quantity` 구조를 정리했다.
- `Knowledge`, `SearchTerm`, `ExpressionTemplate`를 metadata 계층으로 분리했다.
- 관련 파일:
  - [entities.py](../../esgreveal/entities.py)
  - [extensions.py](../../esgreveal/extensions.py)
  - [expressions.py](../../esgreveal/expressions.py)
  - [catalog.py](../../esgreveal/catalog.py)
  - [metadata.py](../../esgreveal/metadata.py)

## 2. 문서 구조 인식

- [document.py](../../esgreveal/document.py)에서 문서 블록을 추출한다.
- 글자 크기와 번호 패턴을 이용해 `section_level`을 추정한다.
- 각 블록에 `outline_path`를 붙여 문서 구조를 추적한다.
- layout backend 우선순위:
  - `GeoLayoutLM`
  - `LayoutLMv3`
  - `layoutparser`
  - `PyMuPDF + OCR fallback`

## 3. GeoLayoutLM 연결

- GeoLayoutLM은 외부 repo를 그대로 호출하는 대신 adapter 구조로 연결했다.
- workspace 전용 adapter:
  - [geolayoutlm_workspace.py](../../esgreveal/adapters/geolayoutlm_workspace.py)
- 참고한 공개 구현:
  - [AlibabaResearch/AdvancedLiterateMachinery](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/blob/main/DocumentUnderstanding/GeoLayoutLM/README.md)

## 4. 표 구조 인식

- [table_structure_analysis.py](../../esgreveal/table_structure_analysis.py)에 표 구조 backend 계층을 추가했다.
- 연결한 backend:
  - `Table Transformer`
  - `LORE-TSR`
  - `PyMuPDF fallback`
- 두 backend의 출력은 공통 `rows / columns / cells` 형식으로 정규화한다.
- 참고한 외부 구현:
  - [microsoft/table-transformer](https://github.com/microsoft/table-transformer)
  - [AlibabaResearch/AdvancedLiterateMachinery - LORE-TSR](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/LORE-TSR)
- 현재 공유본에는 `GeoLayoutLM` 연결에 필요한 부분만 로컬에 남기고, `LORE-TSR`는 로컬 repo 복사본 대신 `ModelScope` 경로를 기본으로 사용한다.

## 5. Structured Knowledge Base

- 논문 구조인 `textual contents / document outlines / table contents`를 반영했다.
- 핵심 구현:
  - [knowledge_base.py](../../esgreveal/knowledge_base.py)
- `textual_contents`는 `mt5` 요약과 `dragonkue/BGE-m3-ko` 임베딩을 사용한다.
- `document_outlines`는 연결된 keyword를 함께 벡터화한다.
- `table_contents`는 `one-to-many mapping`을 적용해 `table_base`와 `table_keyword` record로 분리한다.
- 산출 파일:
  - `records.jsonl`
  - `textual_contents.jsonl`
  - `document_outlines.jsonl`
  - `table_contents.jsonl`
  - `text_vectors.npy`
  - `text_vectors.faiss`
  - `manifest.json`

## 6. CLI와 GUI

- CLI:
  - [cli.py](../../esgreveal/cli.py)
  - `--build-kb` 모드 추가
- GUI:
  - [gui.py](../../esgreveal/gui.py)
  - Knowledge Base 구축, KPI 추출, retrieval 근거 확인, prompt 확인, JSON 저장 지원

## 7. 런타임 정리

- [runtime_env.py](../../esgreveal/runtime_env.py)에서 repo-local cache 기본값을 설정한다.
- PowerShell 세션용 스크립트:
  - [esgreveal_env.ps1](../../scripts/esgreveal_env.ps1)
- 모델 사전 다운로드:
  - [prefetch_models.py](../../scripts/prefetch_models.py)
- 기존 캐시 복사:
  - [sync_existing_caches.ps1](../../scripts/sync_existing_caches.ps1)

## 8. Reranker 선택 기록

- 논문은 1차 cosine retrieval 뒤 `coROM`으로 semantic similarity를 다시 평가한다.
- 현재 구현도 같은 2단계 retrieval 철학을 유지한다.
- 다만 기본 reranker는 한국 ESG 보고서에 맞춰 `dragonkue/bge-reranker-v2-m3-ko`를 사용한다.
- 따라서 현재 구현은 `coROM exact model`이 아니라, 한국어 성능을 우선한 learned reranker 대체 구현이다.

## 9. 관련 문서

- [KNOWLEDGE_BASE_KO.md](../architecture/KNOWLEDGE_BASE_KO.md)
- [LAB_RUNTIME_SETUP_KO.md](../setup/LAB_RUNTIME_SETUP_KO.md)
- [GEOLAYOUTLM_ADAPTER_KO.md](../architecture/GEOLAYOUTLM_ADAPTER_KO.md)
- [TABLE_STRUCTURE_BACKENDS_KO.md](../architecture/TABLE_STRUCTURE_BACKENDS_KO.md)
- [NEXT_STEPS_KO.md](NEXT_STEPS_KO.md)
