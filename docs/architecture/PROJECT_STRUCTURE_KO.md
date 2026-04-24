# 프로젝트 구조 안내

현재 저장소는 "실행 가능한 공유본" 기준으로 정리되어 있다.  
가장 자주 보는 폴더와, 실행하면서 자동 생성되는 폴더를 구분해서 보면 덜 헷갈린다.

## 1. 핵심 폴더

- [esgreveal](../../esgreveal)
  - 본체 코드
  - 전처리, metadata, retrieval, pipeline, GUI 구현이 들어 있다

- [reports](../../reports)
  - 입력용 ESG 보고서 PDF 모음
  - 현재 공유본에는 `2025_kr` 샘플 코퍼스 5건이 들어 있다

- [docs](../)
  - 구조 설명, 사용법, 구현 기록 문서

- [scripts](../../scripts)
  - 환경 설정, 캐시 정리, 모델 준비 스크립트

- [tests](../../tests)
  - 단위 테스트와 스모크 테스트

## 2. 참고 자료

- [references](../../references)
  - 논문과 supplementary 자료를 모아둔 폴더

- [references/papers](../../references/papers)
  - 원 논문 PDF

- [references/supplementary](../../references/supplementary)
  - supplementary 문서

## 3. 외부 구현 보관본

- [external](../../external)
  - 외부 구현을 참고하거나 연결하기 위한 보관 폴더
  - 현재 공유본에는 `GeoLayoutLM` 연결에 필요한 부분과 `table-transformer`가 남아 있다

## 4. 실행 시 자동 생성되는 폴더

아래 폴더는 기본 공유본에 포함하지 않아도 된다. 실행하면 다시 생긴다.

- `outputs`
  - CLI 실행 결과

- `gui_output`
  - GUI에서 저장한 JSON/Markdown 결과

- `esgreveal_knowledge_base`
  - knowledge base 생성 결과

- `.cache`
  - 모델 캐시

- `.venv`
  - 로컬 가상환경

## 5. 루트 파일

- [README.md](../../README.md)
  - 전체 프로젝트 소개와 빠른 시작

- [ESGReveal Setup.cmd](../../ESGReveal%20Setup.cmd)
  - 공유받은 쪽에서 1회 환경 준비

- [ESGReveal GUI.cmd](../../ESGReveal%20GUI.cmd)
  - GUI 실행 진입점

- [pyproject.toml](../../pyproject.toml)
  - 패키지 메타데이터와 의존성 정의

## 6. 보통 작업 순서

1. 입력 PDF는 [reports](../../reports)에서 고른다.
2. 구현 수정은 [esgreveal](../../esgreveal)에서 한다.
3. 설명과 배경은 [docs](../)에서 본다.
4. 실행 결과는 실행 후 생성되는 `outputs` 또는 `gui_output`에서 확인한다.

한 줄로 요약하면, 일상적으로는 `esgreveal / reports / docs / scripts / tests`를 중심으로 보면 된다.
