# 논문 대비 구현 정리

이 문서는 원 논문이 사용한 구성요소와, 현재 프로젝트에서 실제로 어떻게 구현했는지를 한 곳에 모아 정리한 문서다.

## 1. 한눈에 보는 대응표

| 논문 요소 | 논문에서 사용한 방식 | 현재 구현 | 현재 상태 | 이유 / 메모 |
| --- | --- | --- | --- | --- |
| 문서 구조 인식 | `LayoutLMv3`, `GeoLayoutLM` | `GeoLayoutLM -> LayoutLMv3 -> layoutparser -> PyMuPDF+OCR fallback` | 부분 재현 | 논문 철학은 유지했지만, 한국 ESG PDF 실사용을 위해 fallback을 함께 둠 |
| 표 구조 인식 | `Table Transformer`, `LORE-TSR` | `Table Transformer -> LORE-TSR -> PyMuPDF fallback` | 부분 재현 | `LORE-TSR`는 공유본에서 로컬 repo 대신 `ModelScope` 경로를 기본 사용 |
| 요약 | `mt5` | `google/mt5-small` | 직접 사용 | 논문 방향 유지 |
| 임베딩 | `m3e` | `dragonkue/BGE-m3-ko` 기본, 옵션으로 다른 모델 가능 | 한국어 대체 | 논문은 다국어/중국어권 중심, 현재 프로젝트는 한국 ESG 보고서 중심 |
| 벡터 저장 | `FAISS`, `Milvus` | `FAISS`, `Milvus Lite/URI` | 직접 사용 | 논문 방향 유지 |
| 2차 재정렬 | `coROM` | `dragonkue/bge-reranker-v2-m3-ko` | 한국어 대체 | 공개 `coROM` 체크포인트보다 한국어 ESG 문서에 더 적합한 learned reranker 사용 |
| Expert Knowledge | ESG Metadata의 `Knowledge`, `SearchTerm` | metadata에 구현 | 직접 사용 | 논문 구조 반영 |
| Expressions | structured prompt / answer format | metadata의 `ExpressionTemplate` + prompt builder | 직접 사용 | 논문 구조 반영 |
| LLM Agent | RAG 후 KPI별 prompt | retrieval -> rerank -> prompt -> LLM backend | 직접 사용 | Codex/Gemini/Claude backend 지원 |

## 2. 한국어 대응 때문에 바꾼 것

### 2-1. `m3e` 대신 한국어 임베딩

- 논문은 `m3e`를 임베딩 모델로 사용한다.
- 현재 프로젝트는 기본 임베딩을 `dragonkue/BGE-m3-ko`로 둔다.
- 이유:
  - 입력이 한국 ESG 보고서 중심이다.
  - 한국어 retrieval 품질을 우선해야 한다.

관련 위치:

- [runtime_env.py](../../esgreveal/runtime_env.py)
- [knowledge_base.py](../../esgreveal/knowledge_base.py)
- [KNOWLEDGE_BASE_KO.md](KNOWLEDGE_BASE_KO.md)

### 2-2. `coROM` 대신 한국어 reranker

- 논문은 1차 cosine retrieval 뒤 `coROM`으로 semantic similarity를 다시 평가한다.
- 현재 프로젝트는 같은 2단계 retrieval 철학을 유지하되, 기본 reranker를 `dragonkue/bge-reranker-v2-m3-ko`로 둔다.
- 이유:
  - 공개 `coROM` 계열 기본 체크포인트는 중국어/영어 중심이다.
  - 현재 프로젝트는 한국 ESG 보고서를 대상으로 한다.

즉, 현재 구현은 `coROM exact model`이 아니라 **한국어 지원과 한국어 성능을 우선한 learned reranker 대체 구현**이다.

관련 위치:

- [README.md](../../README.md)
- [KNOWLEDGE_BASE_KO.md](KNOWLEDGE_BASE_KO.md)
- [IMPLEMENTATION_LOG_KO.md](../experiments/IMPLEMENTATION_LOG_KO.md)

## 3. GeoLayoutLM은 실제로 쓰고 있나

짧게 말하면, **실제 코드 연결은 되어 있지만 조건부로 동작한다.**

현재 구조:

- adapter 진입점:
  - [geolayoutlm_workspace.py](../../esgreveal/adapters/geolayoutlm_workspace.py)
- 기본 repo 경로:
  - `external/AdvancedLiterateMachinery/DocumentUnderstanding/GeoLayoutLM`

실제로 동작하려면 필요한 것:

- GeoLayoutLM repo 파일
- `ESGREVEAL_GEOLAYOUTLM_CHECKPOINT`
- 관련 Python 의존성

현재 구현은 다음 순서로 쓴다.

1. OCR로 페이지 단어/라인을 분해
2. GeoLayoutLM 입력 형식으로 변환
3. SER checkpoint로 라벨링
4. `HEADER` 계열을 `heading` 신호로 해석

즉, **공개 GeoLayoutLM 본체를 실제로 호출하는 경로는 있다.**

다만 한계도 있다.

- 공개 GeoLayoutLM repo는 FUNSD용 VIE 코드라서 `heading/paragraph/table` 전용 모델은 아니다.
- 그래서 지금은 SER 결과를 이용해 문서 구조 신호로 변환하는 방식이다.
- checkpoint가 없으면 이 경로는 꺼지고, 다음 backend로 fallback한다.

정리하면:

- “실제 GeoLayoutLM을 아예 못 쓴다”는 아님
- “논문과 완전히 같은 목적의 전용 구조 모델로 그대로 쓴다”도 아님
- 현재 상태는 **실제 repo + 실제 checkpoint를 연결해 조건부로 쓰는 adapter 방식**이다

관련 위치:

- [GEOLAYOUTLM_ADAPTER_KO.md](GEOLAYOUTLM_ADAPTER_KO.md)
- [geolayoutlm_workspace.py](../../esgreveal/adapters/geolayoutlm_workspace.py)

## 4. LORE-TSR는 현재 어떻게 쓰나

- 논문은 `LORE-TSR`를 표 구조 인식에 사용한다.
- 현재 구현도 backend 순서에 `LORE-TSR`를 포함한다.
- 다만 공유본에는 로컬 `LORE-TSR` repo 복사본을 남기지 않았고, 기본적으로 `ModelScope` 모델 경로를 사용한다.

즉:

- 논문 철학: 유지
- 로컬 구현 형태: 단순 복사본이 아니라 배포 가능한 경로로 조정

관련 위치:

- [TABLE_STRUCTURE_BACKENDS_KO.md](TABLE_STRUCTURE_BACKENDS_KO.md)
- [table_structure_analysis.py](../../esgreveal/table_structure_analysis.py)

## 5. external 폴더에 남아 있는 것

현재 공유본의 [external](../../external)에는 아래만 남겨두었다.

- `GeoLayoutLM` 연결에 필요한 부분
- `table-transformer`

즉, `external`은 “논문 원안 전체를 다 복사한 곳”이 아니라, **현재 구현에서 실제로 참고하거나 연결하는 외부 구현만 남긴 폴더**다.

## 6. 추천 읽기 순서

1. [PROJECT_STRUCTURE_KO.md](PROJECT_STRUCTURE_KO.md)
2. [KNOWLEDGE_BASE_KO.md](KNOWLEDGE_BASE_KO.md)
3. [GEOLAYOUTLM_ADAPTER_KO.md](GEOLAYOUTLM_ADAPTER_KO.md)
4. [TABLE_STRUCTURE_BACKENDS_KO.md](TABLE_STRUCTURE_BACKENDS_KO.md)
5. [IMPLEMENTATION_LOG_KO.md](../experiments/IMPLEMENTATION_LOG_KO.md)

이 문서를 먼저 보면, “논문 원안에서 무엇을 그대로 썼고 무엇을 한국어 기준으로 바꿨는지”를 가장 빨리 파악할 수 있다.
