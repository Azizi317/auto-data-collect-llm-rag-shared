# 다음 작업 메모

- 값 추출 정확도 개선
  - 최신 연도 우선 선택
  - Scope 1 / Scope 2 / Scope 1+2 / Scope 3 충돌 방지
  - 색인 페이지 숫자 오인 방지

- 문서 유형 분류 안정화
  - ESG 보고서를 `academic_paper`로 잘못 보는 케이스 보정

- 표 구조 품질 비교
  - `Table Transformer`와 `LORE-TSR`를 같은 ESG 표 셋에서 비교

- end-to-end 평가
  - 대표 보고서 묶음에 대해 retrieval, reranking, extraction 결과를 다시 점검
